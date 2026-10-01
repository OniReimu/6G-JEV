"""EXP-2026-003 RQ3 load runner: Poisson schedule, FIFO timing, live fake adapters, rho and refusal."""
from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.edgebench.interpreters.base import Decision, Interpreter
from src.edgebench.ledger import LedgerWriter
from src.ranbench.c2.stats import utilisation
from src.ranbench.c2.streams import stream_from_trace
from src.ranbench import rq3_load
from src.ranbench.rq3_load import (
    fifo_timings,
    load_verified_pool,
    make_load_schedule,
    run_rq3_load,
)

CORPUS = Path("data/ranbench/ranintent-v1")


class FakeAdapter(Interpreter):
    def __init__(self, log: list[str]) -> None:
        super().__init__(name="Fake-RQ3", deployment="reference")
        self.log = log
        self.platform = "test"

    def decide(self, case):
        self.log.append(case.case_id)
        return Decision(
            labels=[dict(case.truth)],
            valid=True,
            http_status=200,
            latency_s=0.25,
            input_tokens=10,
            output_tokens=9,
            cost_usd=0.001,
            resolved_model="fake-rq3-v1",
            provider="fake",
        )


class StatusAdapter(Interpreter):
    def __init__(self, statuses: list[int], *, deployment: str = "hosted", delay_s: float = 0.0) -> None:
        super().__init__(name="Fake-RQ3", deployment=deployment)
        self.statuses = list(statuses)
        self.delay_s = delay_s
        self.platform = "test"

    def decide(self, case):
        if self.delay_s:
            time.sleep(self.delay_s)
        status = self.statuses.pop(0) if self.statuses else 200
        if status != 200:
            return Decision(
                labels=[],
                valid=False,
                http_status=status,
                error_type=f"HTTP_{status}",
                latency_s=self.delay_s,
                provider="fake",
            )
        return Decision(
            labels=[dict(case.truth)],
            valid=True,
            http_status=200,
            latency_s=self.delay_s,
            resolved_model="fake-rq3-v1",
            provider="fake",
        )


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_poisson_schedule_pairs_rates_and_uses_two_complete_pool_passes():
    pool, _, _ = load_verified_pool(CORPUS)
    slow = make_load_schedule(pool, 0.5)
    fast = make_load_schedule(pool, 2.0)
    assert len(slow) == len(fast) == 300
    assert [r["intent_id"] for r in slow] == [r["intent_id"] for r in fast]
    assert np.allclose([r["arrival_s"] for r in slow], np.array([r["arrival_s"] for r in fast]) * 4)
    ids = [r["intent_id"] for r in slow]
    assert len(set(ids[:150])) == len(set(ids[150:])) == 150
    assert sorted(ids[:150]) == sorted(ids[150:])
    assert len(set(ids[:100])) == 100  # C2's replay prefix has one live decision per intent id
    assert {r["pool_pass"] for r in slow[:150]} == {0}
    assert {r["pool_pass"] for r in slow[150:]} == {1}


def test_fifo_queue_waits_match_hand_computed_tiny_schedule():
    rows = fifo_timings(arrivals_s=[0.0, 0.0, 0.5, 1.5], service_s=[1.0, 2.0, 1.0, 1.0], slots=2)
    assert [r["slot"] for r in rows] == [0, 1, 0, 0]
    assert [r["service_start_s"] for r in rows] == [0.0, 0.0, 1.0, 2.0]
    assert [r["queue_wait_s"] for r in rows] == [0.0, 0.0, 0.5, 0.5]
    assert [r["completion_s"] for r in rows] == [1.0, 2.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="non-decreasing"):
        fifo_timings([1.0, 0.0], [1.0, 1.0], slots=2)


def test_rho_definition_and_non_stationary_flag():
    result = utilisation(rate=2.0, service_s=np.array([1.0, 2.0, 3.0]), slots=4)
    assert result == {"rho": 1.0, "non_stationary": True}
    assert utilisation(rate=0.5, service_s=np.array([1.0, 2.0, 3.0]), slots=4) == {
        "rho": 0.25,
        "non_stationary": False,
    }


def test_live_fake_adapters_write_replay_trace_and_existing_ledger_schema(tmp_path):
    log: list[str] = []
    out = tmp_path / "run"
    report = run_rq3_load(
        corpus_dir=CORPUS,
        interpreter_name="Fake-RQ3",
        rate=1_000_000.0,
        out_dir=out,
        slots=2,
        n_intents=6,
        spend_cap_usd=1.0,
        allow_dirty=True,
        interpreters=[FakeAdapter(log), FakeAdapter(log)],
    )
    trace = read_jsonl(out / "trace.jsonl")
    ledger = read_jsonl(out / "ledger.jsonl")
    assert report["complete"] and report["completed_intents"] == report["valid_intents"] == 6
    assert len(log) == len(trace) == len(ledger) == 6
    assert all(r["status"] == "decision_recorded" and r["valid"] and r["policy"] == r["labels"] for r in trace)
    assert all(r["completion_s"] >= r["service_start_s"] >= r["arrival_s"] for r in trace)
    assert all(r["queue_wait_s"] == pytest.approx(r["service_start_s"] - r["arrival_s"]) for r in trace)
    assert report["rho"] == pytest.approx(
        report["rate_per_s"] * np.mean([r["service_latency_s"] for r in trace]) / report["slots"]
    )

    # The trace is accepted without adaptation by the registered RQ3-to-C2 replay function.
    pool, pool_sha, _ = load_verified_pool(CORPUS)
    stream = stream_from_trace(trace, pool, block_s=10.0, key="test-rq3", pool_sha256=pool_sha, n=6)
    expected = [r["intent_id"] for r in sorted(trace, key=lambda r: r["arrival_s"])]
    assert [e["intent_id"] for e in stream["events"]] == expected

    # No RQ3-only timing columns leak into the required existing Edgebench ledger row layout.
    layout = LedgerWriter(tmp_path / "layout.jsonl", "run", "git", "RQ3", "rate")
    expected_keys = set(layout.write_row("Fake-RQ3", "x", 0, Decision(), correct={}, em=False))
    layout.close()
    assert all(set(row) == expected_keys for row in ledger)
    assert all(row["rq"] == "RQ3" and row["model"] == "Fake-RQ3" for row in ledger)


def test_nonempty_output_is_cleanly_refused_before_adapter_call(tmp_path):
    out = tmp_path / "partial"
    out.mkdir()
    sentinel = out / "trace.jsonl"
    sentinel.write_text('{"partial": true}\n', encoding="utf-8")
    log: list[str] = []
    with pytest.raises(RuntimeError, match="Refusing to resume or overwrite non-empty"):
        run_rq3_load(
            corpus_dir=CORPUS,
            interpreter_name="Fake-RQ3",
            rate=1.0,
            out_dir=out,
            slots=2,
            n_intents=2,
            allow_dirty=True,
            interpreters=[FakeAdapter(log), FakeAdapter(log)],
        )
    assert log == [] and sentinel.read_text(encoding="utf-8") == '{"partial": true}\n'


def test_hosted_errors_above_five_percent_make_cli_nonzero_and_keep_every_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(rq3_load, "load_manifest", lambda path=None: {"Fake-RQ3": {"deployment": "hosted"}})
    monkeypatch.setattr(rq3_load, "load_openrouter_key", lambda: "test-key")
    monkeypatch.setattr(
        rq3_load,
        "build_interpreter",
        lambda *args, **kwargs: StatusAdapter([429] * 20),
    )
    out = tmp_path / "hosted-errors"
    rc = rq3_load.main(
        [
            "run",
            "--corpus-dir",
            str(CORPUS),
            "--interpreter",
            "Fake-RQ3",
            "--rate",
            "1000000",
            "--out",
            str(out),
            "--slots",
            "2",
            "--intents",
            "20",
            "--client-location",
            "test",
            "--allow-dirty",
        ]
    )
    report = json.loads((out / "integrity.json").read_text(encoding="utf-8"))
    assert rc == 1
    assert report["completed_intents"] == report["transport_error_rows"] == 20
    assert report["rerun_block_required"] is True and report["complete"] is False
    assert report["degradation_reason"] in report["stop_reason"]
    assert len(read_jsonl(out / "ledger.jsonl")) == len(read_jsonl(out / "trace.jsonl")) == 20


def test_transport_error_rate_exactly_at_threshold_is_complete(tmp_path):
    report = run_rq3_load(
        corpus_dir=CORPUS,
        interpreter_name="Fake-RQ3",
        rate=1_000_000.0,
        out_dir=tmp_path / "boundary",
        slots=1,
        n_intents=20,
        allow_dirty=True,
        client_location="test",
        interpreters=[StatusAdapter([429] + [200] * 19)],
    )
    assert report["transport_error_rate"] == pytest.approx(0.05)
    assert report["rerun_block_required"] is False
    assert report["degradation_reason"] is None and report["stop_reason"] is None
    assert report["complete"] is True


def test_repeated_self_hosted_5xx_trips_reference_circuit_breaker(tmp_path):
    report = run_rq3_load(
        corpus_dir=CORPUS,
        interpreter_name="Fake-RQ3",
        rate=1_000_000.0,
        out_dir=tmp_path / "self-hosted-5xx",
        slots=1,
        n_intents=20,
        allow_dirty=True,
        interpreters=[StatusAdapter([503] * 20, deployment="self-hosted")],
    )
    assert report["completed_intents"] == 3
    assert report["stop_reason"] == "self_hosted_server_error"
    assert report["transport_error_rows"] == 3 and report["rerun_block_required"] is True
    assert report["complete"] is False


def test_slow_call_does_not_delay_later_monotonic_arrivals(tmp_path):
    out = tmp_path / "slow"
    report = run_rq3_load(
        corpus_dir=CORPUS,
        interpreter_name="Fake-RQ3",
        rate=10_000.0,
        out_dir=out,
        slots=1,
        n_intents=4,
        allow_dirty=True,
        client_location="test",
        interpreters=[StatusAdapter([200] * 4, delay_s=0.1)],
    )
    trace = sorted(read_jsonl(out / "trace.jsonl"), key=lambda row: row["event_index"])
    assert report["complete"] is True
    assert [row["observed_arrival_s"] for row in trace] == sorted(row["observed_arrival_s"] for row in trace)
    assert trace[-1]["observed_arrival_s"] < trace[0]["completion_s"]
    assert trace[1]["service_start_s"] >= trace[0]["completion_s"]


def test_hosted_launcher_retries_degraded_cell_in_fresh_block(tmp_path, monkeypatch):
    from scripts import rb_rq3_hosted

    outputs: list[Path] = []

    def fake_run(command, check):
        assert check is False
        out = Path(command[command.index("--out") + 1])
        outputs.append(out)
        out.mkdir(parents=True)
        if len(outputs) == 1:
            (out / "integrity.json").write_text(
                json.dumps({"rerun_block_required": True, "degradation_reason": "too many errors"}),
                encoding="utf-8",
            )
            return SimpleNamespace(returncode=1)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(rb_rq3_hosted.subprocess, "run", fake_run)
    rc = rb_rq3_hosted.main(
        [
            "--out-root",
            str(tmp_path),
            "--client-location",
            "test",
            "--interpreters",
            "Jev-1.13.0",
            "--rates",
            "1",
        ]
    )
    assert rc == 0
    assert [path.name for path in outputs] == ["rate_1__block1", "rate_1__block2"]


def test_rq3_load_imports_without_scipy() -> None:
    """The self-hosted vLLM venv has no scipy; the runner and its utilisation must not need it."""
    import subprocess
    import sys

    code = ("import sys; sys.modules['scipy'] = None; import numpy as np; import src.ranbench.rq3_load as r; "
            "assert r.utilisation(2.0, np.array([1.0, 2.0, 3.0]), 4) == {'rho': 1.0, 'non_stationary': True}")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).resolve().parents[1])


def test_anyjev_ledger_token_count_does_not_occupy_the_slot(tmp_path):
    """The ledger's AnyJev input-token count is bookkeeping: it must stay outside the service window (and so out
    of the slot's busy time and the queue waits) and still reach the ledger."""
    from anyjev.backends.fake import FakeTokenizer

    from src.ranbench.interpreters.anyjev_l0 import AnyJevClient

    class SlowCountTokenizer(FakeTokenizer):
        def encode(self, text, add_special_tokens=False):
            if len(text) > 8:  # whole prompts are only encoded by the ledger count
                time.sleep(0.01)
            return super().encode(text, add_special_tokens=add_special_tokens)

    class UniformBackend:  # label log-probabilities only; the prompts are not parsed
        name = "fake"

        def __init__(self) -> None:
            self.tokenizer = SlowCountTokenizer()

        def next_token_logprobs(self, prompts, token_ids):
            return [np.full(len(ids), -np.log(len(ids) + 1.0)) for ids in token_ids]

    def client() -> AnyJevClient:
        c = AnyJevClient(backend=UniformBackend())
        c.platform = "test"
        return c

    out = tmp_path / "run"
    report = run_rq3_load(
        corpus_dir=CORPUS, interpreter_name="AnyJev-L0", rate=1_000_000.0, out_dir=out, slots=2, n_intents=6,
        spend_cap_usd=1.0, allow_dirty=True, interpreters=[client(), client()],
    )
    trace = read_jsonl(out / "trace.jsonl")
    ledger = read_jsonl(out / "ledger.jsonl")
    assert report["completed_intents"] == 6 and len(ledger) == 6
    gaps = [r["service_latency_s"] - r["adapter_latency_s"] for r in trace]
    assert max(gaps) < 0.05, gaps  # 7a7d5d7 counted ~36 prompts x 10 ms inside the window
    assert all(isinstance(row["input_tokens"], int) and row["input_tokens"] > 0 for row in ledger)


def test_deferred_token_count_failure_keeps_every_ledger_row(tmp_path, monkeypatch):
    from src.ranbench.interpreters import anyjev_l0

    calls = {"n": 0}

    class Flaky:
        def encode(self, text, add_special_tokens=False):
            calls["n"] += 1
            raise RuntimeError("tokenizer broke")

    class Backend:
        name = "fake"
        tokenizer = None

        def next_token_logprobs(self, prompts, token_ids):
            return [np.full(len(ids), -np.log(len(ids) + 1.0)) for ids in token_ids]

    from anyjev.backends.fake import FakeTokenizer

    def client():
        b = Backend()
        b.tokenizer = FakeTokenizer()
        c = anyjev_l0.AnyJevClient(backend=b)
        c.platform = "test"
        return c

    clients = [client(), client()]
    real_decide = anyjev_l0.AnyJevClient.decide

    def decide(self, case):
        d = real_decide(self, case)
        d.count_input_tokens = lambda: Flaky().encode("prompt")
        return d

    monkeypatch.setattr(anyjev_l0.AnyJevClient, "decide", decide)
    out = tmp_path / "run"
    report = run_rq3_load(
        corpus_dir=CORPUS, interpreter_name="AnyJev-L0", rate=1_000_000.0, out_dir=out, slots=2, n_intents=4,
        spend_cap_usd=1.0, allow_dirty=True, interpreters=clients,
    )
    ledger = read_jsonl(out / "ledger.jsonl")
    assert report["completed_intents"] == 4 and len(ledger) == 4 and calls["n"] == 4
    assert all(row["input_tokens"] is None for row in ledger)
