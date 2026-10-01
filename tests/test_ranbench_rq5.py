"""RQ5 arm-(b) protocol, controller, fail-closed action, timing, and replay tests."""
from __future__ import annotations

import json
import socket
import threading
from pathlib import Path

import pytest

from src.ranbench.rq5.controller import serve_connection
from src.ranbench.rq5.protocol import (
    RQ5_RULES,
    Rq5Result,
    application_time,
    apply_action,
    decision_state,
    normalise_actions,
    response_schema,
    rq5_context,
    rq5_questions,
    tick_is_skipped,
)


@pytest.fixture
def snapshot() -> dict:
    return {
        "tick_s": 1.0,
        "window_s": 1.0,
        "step": 2,
        "band": 10,
        "rows": [
            {"cell": 1, "class": "video", "throughput_kbps": 4800.0, "p95_delay_ms": 10.0,
             "prb_share": 0.6, "base_priority": 30, "current_priority": 30,
             "target_type": "throughput_floor_kbps", "target_value": 5000.0},
            {"cell": 1, "class": "xr", "throughput_kbps": 1900.0, "p95_delay_ms": 24.0,
             "prb_share": 0.4, "base_priority": 50, "current_priority": 48,
             "target_type": "delay_p95_ms", "target_value": 20.0},
        ],
    }


def test_rendering_parity_and_dynamic_schema(snapshot):
    context = rq5_context(snapshot)
    assert context["rules"] == RQ5_RULES
    assert decision_state(snapshot) == context["rules"] + "\n\n" + context["case"]
    questions = rq5_questions(snapshot)
    assert list(questions) == ["cell_1__video", "cell_1__xr"]
    assert all(question["type"] == "choice" for question in questions.values())
    assert all(list(question["criteria"]) == ["raise", "hold", "lower"] for question in questions.values())
    schema = response_schema(snapshot)
    assert schema["properties"]["actions"]["minItems"] == 2
    assert schema["properties"]["actions"]["maxItems"] == 2


def test_invalid_or_missing_field_becomes_counted_hold(snapshot):
    result = normalise_actions(snapshot, {"actions": [
        {"cell": 1, "class": "video", "action": "raise"},
        {"cell": 1, "class": "xr", "action": "boost"},
    ]})
    assert result.actions == [
        {"cell": 1, "class": "video", "action": "raise"},
        {"cell": 1, "class": "xr", "action": "hold"},
    ]
    assert result.invalid_fields == 1


def test_timing_skip_direction_and_active_policy_clamp():
    assert application_time(1.0, 0.995) == pytest.approx(2.0)
    assert tick_is_skipped(2.0, 2.000001)
    assert not tick_is_skipped(2.0, 2.0)
    assert apply_action(30, 30, "raise", 2, 10) == 28
    assert apply_action(30, 30, "lower", 2, 10) == 32
    assert apply_action(20, 30, "raise", 2, 10) == 20
    assert apply_action(40, 30, "lower", 2, 10) == 40
    assert apply_action(2, 5, "raise", 4, 10) == 1


class FakeInterpreter:
    name = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def decide_rq5(self, snapshot: dict) -> Rq5Result:
        self.calls += 1
        return Rq5Result(
            actions=[
                {"cell": row["cell"], "class": row["class"], "action": "raise"}
                for row in snapshot["rows"]
            ],
            invalid_fields=0,
        )


def _exchange(out: Path, messages: list[dict], interpreter=None, replay_dir=None) -> list[dict]:
    server, client = socket.socketpair()
    failures = []

    def run() -> None:
        try:
            with server:
                serve_connection(server, out, interpreter=interpreter, replay_dir=replay_dir)
        except Exception as exc:  # noqa: BLE001 - re-raised in the test thread
            failures.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    replies = []
    with client, client.makefile("r", encoding="utf-8") as reader, client.makefile("w", encoding="utf-8") as writer:
        for message in messages:
            writer.write(json.dumps(message) + "\n")
            writer.flush()
            replies.append(json.loads(reader.readline()))
    thread.join(timeout=2)
    assert not thread.is_alive() and not failures
    return replies


def test_fake_live_skip_and_replay_identity(tmp_path, snapshot):
    live = tmp_path / "live"
    live.mkdir()
    fake = FakeInterpreter()
    messages = [
        {"type": "snapshot", **snapshot},
        {"type": "skipped", "tick_s": 2.0, "pending_until_s": 2.5},
    ]
    live_replies = _exchange(live, messages, interpreter=fake)
    assert fake.calls == 1
    assert live_replies[1] == {"type": "ack", "tick_s": 2.0}
    replay = tmp_path / "replay"
    replay.mkdir()
    replay_replies = _exchange(replay, messages, replay_dir=live)
    assert replay_replies == live_replies
    assert (replay / "snapshots.jsonl").read_bytes() == (live / "snapshots.jsonl").read_bytes()
    live_decision = json.loads((live / "decisions.jsonl").read_text().splitlines()[0])
    replay_decision = json.loads((replay / "decisions.jsonl").read_text().splitlines()[0])
    for key in ("tick_s", "latency_s", "invalid_fields", "actions"):
        assert replay_decision[key] == live_decision[key]


def test_rq5_row_runs_on_its_arm_a_inputs(tmp_path):
    from src.ranbench.c2.matrix import INTERPRETERS, generate_matrix, write_matrix
    from src.ranbench.c2.queue import load_rows
    from src.ranbench.rq5.controller import resolve_row_inputs

    path = tmp_path / "matrix.csv"
    write_matrix(generate_matrix(tmp_path / "inputs"), path)
    rows = load_rows(path)
    by_id = {row["run_id"]: row for row in rows}
    rq5 = [row for row in rows if row["rqs"] == "RQ5"]
    assert sorted(row["interpreter"] for row in rq5) == sorted(INTERPRETERS)
    for row in rq5:
        inputs = resolve_row_inputs(row, by_id)
        a_row = by_id[row["reused_arm_a_run_id"]]
        assert inputs["schedule"] == Path(a_row["schedule_file"])
        assert inputs["stream"] == Path(a_row["stream_file"])
        assert inputs["config"] == Path(a_row["config_file"])
        assert inputs["sim_time"] == float(a_row["simulated_seconds"])
        assert by_id[row["reused_arm_c_run_id"]]["config_file"] == a_row["config_file"]
    bad = dict(rq5[0], reused_arm_a_run_id=rq5[1]["reused_arm_a_run_id"])
    with pytest.raises(ValueError):
        resolve_row_inputs(bad, by_id)


def test_fake_interpreter_is_seeded_per_tick(snapshot, monkeypatch):
    from src.ranbench.rq5 import interpreters as rq5i

    monkeypatch.setattr(rq5i.time, "sleep", lambda _s: None)
    first = rq5i.FakeRq5Interpreter(seed=3).decide_rq5(snapshot)
    again = rq5i.FakeRq5Interpreter(seed=3).decide_rq5(snapshot)
    assert first.actions == again.actions
    assert [(a["cell"], a["class"]) for a in first.actions] == [(1, "video"), (1, "xr")]


def test_chat_json_budget_covers_full_grid(monkeypatch):
    from src.ranbench.interpreters.chat_json import RanChatJsonClient
    from src.ranbench.rq5 import interpreters as rq5i

    rows = [
        {"cell": c, "class": k, "throughput_kbps": None, "p95_delay_ms": None, "prb_share": 0.0,
         "base_priority": 50, "current_priority": 50, "target_type": "delay_p95_ms", "target_value": 20.0}
        for c in range(1, 22) for k in ("video", "xr", "iot", "be")
    ]
    snap = {"tick_s": 1.0, "window_s": 1.0, "step": 2, "band": 10, "rows": rows}
    client = RanChatJsonClient(name="x", model="m", provider_slug="p")
    sent = {}

    def fake_post(payload):
        sent.update(payload)
        body = {"model": "m", "usage": {"completion_tokens_details": {"reasoning_tokens": 0}},
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(
                    {"actions": [{"cell": r["cell"], "class": r["class"], "action": "hold"} for r in rows]})}}]}
        return 200, json.dumps(body), 0.1, 0.0, 0.1, None

    monkeypatch.setattr(client, "post", fake_post)
    result = rq5i.query_chat_json(client, snap)
    assert sent["max_tokens"] >= 64 + 20 * len(rows)
    assert sent["temperature"] == 0 and sent["response_format"]["json_schema"]["strict"] is True
    assert result.invalid_fields == 0 and result.error_type is None and len(result.actions) == 84


@pytest.mark.parametrize(
    ("adapter_latency_s", "inherited_timeout_s", "expected_status", "expected_action", "expected_latency_s"),
    [(30.1, 120.0, "skipped", "hold", 30.0), (29.9, 60.0, "decision", "raise", 29.9)],
)
def test_frozen_deadline_skips_only_calls_past_30_seconds(
    tmp_path, monkeypatch, snapshot, adapter_latency_s, inherited_timeout_s,
    expected_status, expected_action, expected_latency_s,
):
    from src.ranbench.rq5 import controller
    from src.ranbench.rq5.protocol import RQ5_DEADLINE_S

    class TimedInterpreter(FakeInterpreter):
        timeout_s = inherited_timeout_s

        def decide_rq5(self, snap: dict) -> Rq5Result:
            result = super().decide_rq5(snap)
            return Rq5Result(
                result.actions, result.invalid_fields, result.error_type,
                {"adapter_latency_s": adapter_latency_s, "post_timeout_wait_s": 7.0},
            )

    out = tmp_path / expected_status
    out.mkdir()
    monkeypatch.setattr(controller, "build_interpreter", lambda *_args, **_kwargs: TimedInterpreter())
    interpreter = controller.make_interpreter("configured", None, None)
    reply = _exchange(out, [{"type": "snapshot", **snapshot}], interpreter=interpreter)[0]
    decision = json.loads((out / "decisions.jsonl").read_text())
    call = json.loads((out / "calls.jsonl").read_text())
    from src.ranbench.rq5.controller import _manifest_metrics
    metrics = _manifest_metrics([decision])

    assert RQ5_DEADLINE_S == 30.0 and interpreter.timeout_s == RQ5_DEADLINE_S
    assert decision["status"] == expected_status
    assert reply["latency_s"] == expected_latency_s == call["latency_s"]
    assert {item["action"] for item in reply["actions"]} == {expected_action}
    assert decision.get("timed_out", False) is (expected_status == "skipped")
    assert metrics["rq5_deadline_s"] == 30.0
    assert metrics["rq5_timeouts"] == metrics["rq5_skipped"] == (expected_status == "skipped")
    if expected_status == "skipped":
        replay = tmp_path / "replay"
        replay.mkdir()
        assert _exchange(replay, [{"type": "snapshot", **snapshot}], replay_dir=out) == [reply]
        assert json.loads((replay / "decisions.jsonl").read_text())["timed_out"] is True


def test_attempts_archive_incomplete_work_promote_atomically_and_skip_complete(tmp_path):
    from src.ranbench.rq5.controller import prepare_attempt, promote_attempt

    run_dir = tmp_path / "rq5-run"
    stale = Path(str(run_dir) + ".attempt-1")
    stale.mkdir()
    (stale / "partial.txt").write_text("keep", encoding="utf-8")

    attempt = prepare_attempt(run_dir)
    assert attempt == Path(str(run_dir) + ".attempt-2")
    assert not stale.exists()
    archived = Path(str(stale) + ".incomplete")
    assert (archived / "partial.txt").read_text(encoding="utf-8") == "keep"

    attempt.mkdir()
    (attempt / "manifest.json").write_text(
        json.dumps({"complete": True, "status": "complete"}), encoding="utf-8"
    )
    promote_attempt(attempt, run_dir)
    assert run_dir.is_dir() and not attempt.exists()
    assert prepare_attempt(run_dir) is None

    incomplete_final = tmp_path / "legacy-run"
    incomplete_final.mkdir()
    with pytest.raises(RuntimeError, match="complete manifest"):
        promote_attempt(incomplete_final, tmp_path / "must-not-exist")
    assert prepare_attempt(incomplete_final) == Path(str(incomplete_final) + ".attempt-2")
    assert Path(str(incomplete_final) + ".attempt-1.incomplete").is_dir()



def test_controller_error_is_reported_when_ns3_aborts(tmp_path):
    """A failing interpreter closes the socket and ns-3 aborts; the error names the controller's exception."""
    import sys
    import textwrap

    from src.ranbench.rq5.controller import run_simulator

    fake_ns3 = tmp_path / "fake-ns3"
    fake_ns3.write_text(textwrap.dedent(f"""\
        #!{sys.executable}
        import json, socket, sys
        path = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--rq5Socket=")][0]
        s = socket.socket(socket.AF_UNIX); s.connect(path)
        s.sendall((json.dumps({{"type": "snapshot", "tick_s": 1.0, "step": 2, "band": 10, "rows": []}}) + "\\n").encode())
        sys.exit(0 if s.makefile().readline() else 134)
        """))
    fake_ns3.chmod(0o755)
    for name in ("config.json", "schedule.csv"):
        (tmp_path / name).write_text("{}" if name.endswith("json") else "t_s,scope,class,priority\n")

    class Broken:
        name = "broken"

        def decide_rq5(self, snapshot):
            raise OSError("tokenizer path not found")

    with pytest.raises(RuntimeError, match=r"RQ5 controller failed: OSError\('tokenizer path not found'\).*ns-3 exited 134"):
        run_simulator(binary=fake_ns3, config=tmp_path / "config.json", schedule=tmp_path / "schedule.csv",
                      run_dir=tmp_path / "run", rng_run=1, sim_time=1.0, interpreter=Broken(), replay_dir=None)


def test_anyjev_bookkeeping_is_excluded_from_latency(tmp_path, monkeypatch):
    """Client setup (tokenizer load) and the post-call token count are reported as untimed_s and do not delay
    the action."""
    import time as _time
    from types import SimpleNamespace

    from src.ranbench.interpreters.anyjev_l0 import AnyJevClient
    from src.ranbench.rq5 import controller
    from src.ranbench.rq5.interpreters import query_anyjev

    snap = {"tick_s": 1.0, "window_s": 1.0, "step": 2, "band": 10, "rows": [
        {"cell": 1, "class": "video", "throughput_kbps": 1.0, "p95_delay_ms": 1.0, "prb_share": 0.1,
         "base_priority": 50, "current_priority": 50, "target_type": "throughput_floor_kbps", "target_value": 5.0}]}
    client = AnyJevClient(backend_url="http://127.0.0.1:1")

    class Tok:
        def encode(self, prompt, add_special_tokens=False):
            _time.sleep(0.15)
            return [1, 2]

    def slow_setup():
        _time.sleep(0.2)
        client._recorder = SimpleNamespace(prompts=[], fallbacks=0, tokenizer=Tok())
        client._decider = SimpleNamespace(decide=lambda state, qs: (
            client._recorder.prompts.append("p") or [SimpleNamespace(probs=[0.1, 0.8, 0.1]) for _ in qs]))

    monkeypatch.setattr(client, "_setup", slow_setup)
    result = query_anyjev(client, snap)
    assert result.error_type is None and result.metadata["untimed_s"] >= 0.35
    assert result.metadata["adapter_latency_s"] < 0.1

    monkeypatch.setattr(controller, "query_interpreter", lambda _i, s: (_time.sleep(0.4), result)[1])
    left, right = socket.socketpair()
    worker = threading.Thread(target=serve_connection, args=(right, tmp_path, SimpleNamespace(name="AnyJev-L0")))
    worker.start()
    left.sendall((json.dumps({"type": "snapshot", **snap}) + "\n").encode())
    reply = json.loads(left.makefile("r").readline())
    left.close()
    worker.join(5)
    assert reply["latency_s"] < 0.2
