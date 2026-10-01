"""HTTP transport with thread-local persistent connections, host allow-list, and defensive redaction."""
from __future__ import annotations

import hashlib
import http.client
import json
import threading
import time
from typing import Any
import urllib.parse

ALLOWED_HOSTS = frozenset({"openrouter.ai", "localhost", "127.0.0.1"})
# Loopback hosts are the self-hosted, one-request-at-a-time servers.
SELF_HOSTED_HOSTS = frozenset({"localhost", "127.0.0.1"})
POST_TIMEOUT_HEALTH_WAIT_S = 600.0
# Failures of a reused keep-alive connection that the peer closed while idle. While sending: reset, broken pipe,
# abort. While reading the status line only RemoteDisconnected, which http.client raises when the line is empty
# (no response byte arrived); a reset in the middle of the status line is not retried. Timeouts are never retried:
# the server may still be working on the request.
_STALE_SEND_ERRORS = (ConnectionResetError, BrokenPipeError, ConnectionAbortedError)

_local = threading.local()


def _get_connections() -> dict[tuple[str, str, int], http.client.HTTPConnection]:
    if not hasattr(_local, "connections"):
        _local.connections = {}
    return _local.connections


def get_connection(base_url: str, timeout_s: float = 30.0) -> http.client.HTTPConnection:
    parsed = urllib.parse.urlsplit(base_url)
    host = parsed.hostname
    if not host or host not in ALLOWED_HOSTS:
        raise ValueError(
            f"Host '{host}' is not in edgebench allowed hosts: {sorted(ALLOWED_HOSTS)}"
        )
    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError(f"Unsupported scheme: {scheme}")
    port = parsed.port or (443 if scheme == "https" else 80)
    key = (scheme, host, port)

    conns = _get_connections()
    conn = conns.get(key)
    if conn is None:
        if scheme == "https":
            conn = http.client.HTTPSConnection(host, port=port, timeout=timeout_s)
        else:
            conn = http.client.HTTPConnection(host, port=port, timeout=timeout_s)
        conns[key] = conn
    return conn


def reset_send_marker() -> None:
    """Clear this thread's record of the last request send (and of the last post-timeout wait)."""
    _local.last_send = None
    _local.post_timeout_wait_s = None
    _local.post_timeout_healthy = None
    _local.retried = False
    _local.first_error = None
    _local.failed_attempt_s = None


def get_retry_info() -> tuple[bool, str | None, float | None]:
    """(retried, first_error, failed_attempt_s) of this thread's last request since reset: whether a stale reused
    connection was retried once, the first attempt's error class, and the seconds from its send to the retry's send
    (callers timing the whole call subtract this so latency runs from the retry)."""
    return (getattr(_local, "retried", False), getattr(_local, "first_error", None),
            getattr(_local, "failed_attempt_s", None))


def get_post_timeout_wait() -> float | None:
    """Seconds this thread waited on /health after its last timed-out self-hosted request since reset."""
    return getattr(_local, "post_timeout_wait_s", None)


def get_post_timeout_healthy() -> bool | None:
    """Whether /health answered 200 within the wait after this thread's last timed-out request (None: no wait)."""
    return getattr(_local, "post_timeout_healthy", None)


def wait_for_health(
    base_url: str, max_wait_s: float = POST_TIMEOUT_HEALTH_WAIT_S, poll_s: float = 1.0
) -> tuple[float, bool]:
    """Untimed, blocking GET /health on a fresh connection until it answers 200; returns (seconds waited, healthy).

    A single-threaded server answers only after it has finished the request the client abandoned, so the
    next timed request does not queue behind it. Gives up after max_wait_s, or at once when the server
    refuses connections (it is gone; the next request reports that).
    """
    parsed = urllib.parse.urlsplit(base_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    t0 = time.perf_counter()
    healthy = False
    while True:
        remaining = max_wait_s - (time.perf_counter() - t0)
        if remaining <= 0:
            break
        conn = http.client.HTTPConnection(host, port=port, timeout=remaining)
        try:
            conn.request("GET", "/health")
            response = conn.getresponse()
            response.read()
            if response.status == 200:
                healthy = True
                break
        except ConnectionRefusedError:
            break
        except Exception:
            pass
        finally:
            conn.close()
        time.sleep(min(poll_s, max(0.0, max_wait_s - (time.perf_counter() - t0))))
    return time.perf_counter() - t0, healthy


def get_send_marker() -> tuple[float, float] | None:
    """(t_send_wall, t_send_perf) of this thread's last request since reset, or None if none was sent."""
    return getattr(_local, "last_send", None)


def close_thread_connections() -> None:
    conns = _get_connections()
    for conn in list(conns.values()):
        try:
            conn.close()
        except Exception:
            pass
    conns.clear()


def make_request(
    base_url: str,
    path: str,
    payload: dict[str, Any],
    api_key: str | None = None,
    extra_headers: dict[str, str] | None = None,
    timeout_s: float = 30.0,
) -> tuple[int | None, str, float, float, float, str | None, dict[str, str]]:
    """Execute a single HTTP request with keep-alive, no redirects, and at most one retry.

    A REUSED keep-alive connection that fails before any response byte (send error, or the status line never
    arrives: ConnectionReset / BrokenPipe / ConnectionAborted on send, RemoteDisconnected on an empty status
    line) is closed and the request is
    retried ONCE on a fresh connection; latency and send times are then taken from the retry, and
    get_retry_info() reports (True, first error class). A fresh connection's failure, a timeout, and any failure
    after the status line are never retried.

    Returns:
        (http_status, raw_body_redacted, latency_s, t_send_wall, t_recv_wall, error_type, response_headers)
    """
    body_bytes = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-OpenRouter-Title": "Edgebench evaluation",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if extra_headers:
        headers.update(extra_headers)

    parsed = urllib.parse.urlsplit(base_url)
    scheme = parsed.scheme.lower()
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if scheme == "https" else 80)
    conn_key = (scheme, host, port)

    conn = get_connection(base_url, timeout_s=timeout_s)
    _local.retried = False
    _local.first_error = None
    _local.failed_attempt_s = None
    first_start: float | None = None

    for attempt in (1, 2):
        # An open socket before the request means the pooled connection is being reused.
        reused = conn.sock is not None
        t_send_wall = time.time()
        t_start = time.perf_counter()
        if first_start is None:
            first_start = t_start
        else:
            _local.failed_attempt_s = t_start - first_start
        _local.last_send = (t_send_wall, t_start)
        status: int | None = None
        raw_response = ""
        error_type: str | None = None
        response_headers: dict[str, str] = {}
        stage = "send"
        retry = False

        try:
            conn.request("POST", path, body=body_bytes, headers=headers)
            stage = "status"
            response = conn.getresponse()
            stage = "body"
            raw_response = response.read().decode("utf-8", errors="replace")
            status = response.status
            response_headers = {k.lower(): v for k, v in response.getheaders()}
        except Exception as exc:
            error_type = type(exc).__name__
            try:
                conn.close()
            except Exception:
                pass
            conns = _get_connections()
            conns.pop(conn_key, None)
            retry = attempt == 1 and reused and (
                (stage == "send" and isinstance(exc, _STALE_SEND_ERRORS))
                or (stage == "status" and isinstance(exc, http.client.RemoteDisconnected))
            )
        t_recv_wall = time.time()
        latency_s = time.perf_counter() - t_start
        if not retry:
            break
        _local.retried = True
        _local.first_error = error_type
        conn = get_connection(base_url, timeout_s=timeout_s)

    if error_type == "TimeoutError" and host in SELF_HOSTED_HOSTS:
        # Outside the timed window: let the server finish the abandoned request before the next one.
        _local.post_timeout_wait_s, _local.post_timeout_healthy = wait_for_health(
            base_url, max_wait_s=POST_TIMEOUT_HEALTH_WAIT_S
        )

    # Defensive redaction: redact api_key from raw_response if present
    if api_key and api_key in raw_response:
        raw_response = raw_response.replace(api_key, "[REDACTED]")

    return status, raw_response, latency_s, t_send_wall, t_recv_wall, error_type, response_headers


def post_json_once(base_url: str, path: str, payload: dict[str, Any], timeout_s: float) -> tuple[int, str]:
    """Single POST on a fresh connection outside the keep-alive pool (untimed control calls, e.g. /warmup).

    Returns (http_status, body). Connection errors propagate to the caller.
    """
    parsed = urllib.parse.urlsplit(base_url)
    host = parsed.hostname
    if not host or host not in ALLOWED_HOSTS:
        raise ValueError(f"Host '{host}' is not in edgebench allowed hosts: {sorted(ALLOWED_HOSTS)}")
    scheme = parsed.scheme.lower()
    port = parsed.port or (443 if scheme == "https" else 80)
    conn_cls = http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection
    conn = conn_cls(host, port=port, timeout=timeout_s)
    try:
        conn.request("POST", path, body=json.dumps(payload).encode("utf-8"),
                     headers={"Content-Type": "application/json", "Accept": "application/json"})
        response = conn.getresponse()
        return response.status, response.read().decode("utf-8", errors="replace")
    finally:
        conn.close()
