"""make_request: one retry only for a REUSED keep-alive connection that fails before any response byte."""
from __future__ import annotations

import http.client
import json
import socket
import threading
import time

import pytest

from src.edgebench.interpreters import transport


class _Response:
    def __init__(self, body: bytes = b'{"ok":true}', read_error: Exception | None = None) -> None:
        self.status = 200
        self._body = body
        self._read_error = read_error

    def read(self) -> bytes:
        if self._read_error is not None:
            raise self._read_error
        return self._body

    def getheaders(self) -> list[tuple[str, str]]:
        return [("Content-Type", "application/json")]


class _FakeConnection:
    """Scripted stand-in for http.client.HTTPConnection. Each instance takes the next script from `plans`:
    ("ok",) | ("send", exc) | ("status", exc, delay_s) | ("body", exc)."""

    plans: list[tuple] = []
    created: list["_FakeConnection"] = []

    def __init__(self, host: str, port: int | None = None, timeout: float | None = None) -> None:
        self.plan = list(type(self).plans.pop(0))
        self.sock = None
        self.requests = 0
        self.closed = False
        type(self).created.append(self)

    def request(self, method, path, body=None, headers=None) -> None:
        self.requests += 1
        if self.plan[0] == "send":
            raise self.plan[1]
        self.sock = object()

    def getresponse(self) -> _Response:
        kind = self.plan[0]
        if kind == "status":
            time.sleep(self.plan[2])
            raise self.plan[1]
        if kind == "body":
            return _Response(read_error=self.plan[1])
        return _Response()

    def close(self) -> None:
        self.closed = True
        self.sock = None


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(transport.http.client, "HTTPConnection", _FakeConnection)
    _FakeConnection.plans, _FakeConnection.created = [], []
    transport.close_thread_connections()
    transport.reset_send_marker()
    yield _FakeConnection
    transport.close_thread_connections()


URL = "http://127.0.0.1:18999"


def _post():
    return transport.make_request(URL, "/v1/x", {"a": 1}, timeout_s=30.0)


def _make_reused(fake, next_plan: tuple) -> _FakeConnection:
    """First call succeeds and leaves an open pooled connection; its next request follows next_plan."""
    fake.plans.append(("ok",))
    assert _post()[5] is None
    conn = fake.created[-1]
    assert conn.sock is not None
    conn.plan = list(next_plan)
    return conn


def test_stale_reused_connection_is_retried_once_and_timed_from_the_retry(fake):
    stale = _make_reused(fake, ("status", http.client.RemoteDisconnected("closed"), 0.3))
    fake.plans.append(("ok",))
    status, raw, latency_s, t_send, _t_recv, error, _ = _post()
    assert (status, error) == (200, None) and raw == '{"ok":true}'
    assert stale.closed and len(fake.created) == 2 and fake.created[1].requests == 1
    retried, first_error, failed_attempt_s = transport.get_retry_info()
    assert retried is True and first_error == "RemoteDisconnected"
    assert latency_s < 0.2 and failed_attempt_s >= 0.3
    assert transport.get_send_marker()[0] == t_send


@pytest.mark.parametrize("exc", [BrokenPipeError(), ConnectionResetError(), ConnectionAbortedError()])
def test_send_side_failures_of_a_reused_connection_are_retried(fake, exc):
    _make_reused(fake, ("send", exc))
    fake.plans.append(("ok",))
    assert _post()[5] is None
    assert transport.get_retry_info()[:2] == (True, type(exc).__name__)


def test_fresh_connection_failure_is_not_retried(fake):
    fake.plans.append(("status", http.client.RemoteDisconnected("closed"), 0.0))
    status, _raw, _lat, _ts, _tr, error, _ = _post()
    assert (status, error) == (None, "RemoteDisconnected")
    assert len(fake.created) == 1 and transport.get_retry_info() == (False, None, None)


def test_failure_after_the_status_line_is_not_retried(fake):
    _make_reused(fake, ("body", ConnectionResetError()))
    status, _raw, _lat, _ts, _tr, error, _ = _post()
    assert (status, error) == (None, "ConnectionResetError")
    assert len(fake.created) == 1 and transport.get_retry_info()[0] is False


def test_reset_while_reading_the_status_line_is_not_retried(fake):
    # Only RemoteDisconnected proves the status line was empty; a reset may follow partial status-line bytes.
    _make_reused(fake, ("status", ConnectionResetError(), 0.0))
    fake.plans.append(("ok",))
    assert _post()[5] == "ConnectionResetError"
    assert len(fake.created) == 1 and transport.get_retry_info()[0] is False


def test_timeout_on_a_reused_connection_is_not_retried(fake, monkeypatch):
    _make_reused(fake, ("status", TimeoutError(), 0.0))
    fake.plans.append(("ok",))
    # A loopback timeout triggers the untimed health wait; keep it instant.
    monkeypatch.setattr(transport, "wait_for_health", lambda *a, **k: (0.0, True))
    error = _post()[5]
    assert error == "TimeoutError" and len(fake.created) == 1 and transport.get_retry_info()[0] is False


def test_retry_happens_at_most_once(fake):
    _make_reused(fake, ("status", http.client.RemoteDisconnected("closed"), 0.0))
    fake.plans.append(("status", http.client.RemoteDisconnected("again"), 0.0))
    status, _raw, _lat, _ts, _tr, error, _ = _post()
    assert (status, error) == (None, "RemoteDisconnected")
    assert len(fake.created) == 2 and transport.get_retry_info()[:2] == (True, "RemoteDisconnected")


def test_retry_info_resets_on_the_next_call(fake):
    _make_reused(fake, ("status", http.client.RemoteDisconnected("closed"), 0.0))
    fake.plans.append(("ok",))
    _post()
    assert transport.get_retry_info()[0] is True
    _post()  # reuses the fresh connection successfully
    assert transport.get_retry_info() == (False, None, None)


def test_rq5_controller_latency_excludes_the_failed_attempt(tmp_path, monkeypatch):
    from src.ranbench.rq5 import controller
    from src.ranbench.rq5.protocol import Rq5Result

    def fake_query(_interp, snapshot):
        time.sleep(0.3)  # the failed first attempt
        return Rq5Result([], 0, None, {"adapter_latency_s": 0.01, "retried": True,
                                      "first_error": "RemoteDisconnected", "failed_attempt_s": 0.3})

    monkeypatch.setattr(controller, "query_interpreter", fake_query)

    class Interp:
        name = "x"

    left, right = socket.socketpair()
    worker = threading.Thread(target=controller.serve_connection, args=(right, tmp_path, Interp()))
    worker.start()
    left.sendall(b'{"type":"snapshot","tick_s":1.0,"band":10,"step":2,"rows":[]}\n')
    reply = left.makefile("r").readline()
    left.close()
    worker.join(5)
    call = json.loads((tmp_path / "calls.jsonl").read_text().splitlines()[0])
    assert call["retried"] is True and call["latency_s"] < 0.1 and call["wall_latency_s"] >= 0.3
    assert json.loads(reply)["latency_s"] < 0.1
    assert controller._manifest_metrics(
        [json.loads(line) for line in (tmp_path / "decisions.jsonl").read_text().splitlines()]
    )["rq5_retried_calls"] == 1
