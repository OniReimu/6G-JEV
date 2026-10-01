"""Offline tests for the EXP-2026-003 C3 real-stack driver."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.edgebench.interpreters.base import Decision, Interpreter
from src.ranbench.c3.a1 import A1Client, HttpResult
from src.ranbench.c3.driver import (
    ClockBracket,
    ClockPoint,
    ClockProbeSet,
    Completion,
    DockerClockProber,
    LogCompletionWaiter,
    make_policy_id,
    run_c3,
)
from src.ranbench.c3.mapping import CLASS_TO_SLICE, PRIORITY_TO_QUOTA, control_for_policy
from src.ranbench.c3.parsers import (
    first_gnb_acknowledge,
    first_kpm_change,
    parse_gnb_control_exchanges,
    parse_a1_put_acknowledgements,
    parse_timestamp,
    parse_xapp_controls,
)
from src.ranbench.c3.xapp import policy_belongs_to_run, submit_control


def test_fixed_mapping_stays_inside_spike_b2_controls() -> None:
    assert set(CLASS_TO_SLICE.values()) == {0}
    assert PRIORITY_TO_QUOTA == {"low": 5, "normal": 30, "high": 100, "critical": 100, "unspecified": 100}
    assert set(PRIORITY_TO_QUOTA.values()) == {5, 30, 100}
    control = control_for_policy({"action": "deprioritise", "class": "factory_control", "priority": "normal"})
    assert control.to_dict() == {
        "slice_id": 0,
        "min_prb_ratio": 0,
        "max_prb_ratio": 30,
        "dedicated_prb_ratio": 100,
    }
    with pytest.raises(ValueError, match="revert_default"):
        control_for_policy({"action": "revert_default", "class": "best_effort", "priority": "high"})


def test_fixture_log_parsers_match_a1_xapp_and_spike_b2_gnb_lines() -> None:
    a1_line = json.dumps({
        "event": "a1_put_ack",
        "policy_id": "c3-00-i1",
        "status": 201,
        "t_request": "2026-09-25T11:39:32.050000Z",
        "t_utc": "2026-09-25T11:39:32.055000Z",
        "elapsed_s": 0.005,
    })
    xapp_line = json.dumps({
        "event": "control_sent",
        "policy_id": "c3-00-i1",
        "intent_id": "i1",
        "t_seen": "2026-09-25T11:39:32.060000Z",
        "t_control_sent": "2026-09-25T11:39:32.062000Z",
        "control": {"slice_id": 0, "max_prb_ratio": 5},
    })
    gnb_line = "2026-09-25T11:39:32.067332 [E2-DU   ] [I] Sending E2 RIC Control Acknowledge"
    assert parse_a1_put_acknowledgements("noise\n" + a1_line)[0]["elapsed_s"] == pytest.approx(0.005)
    assert parse_xapp_controls(xapp_line)[0]["policy_id"] == "c3-00-i1"
    sent = parse_timestamp("2026-09-25T11:39:32.062000Z")
    assert first_gnb_acknowledge(gnb_line, sent) - sent == pytest.approx(0.005332, abs=1e-8)


def test_gnb_exchange_parser_pairs_serial_requests_and_rejects_overlap() -> None:
    request = "2026-09-25T11:39:32.064483 [E2-DU   ] [I] Received RIC Control Request"
    ack = "2026-09-25T11:39:32.067332 [E2-DU   ] [I] Sending E2 RIC Control Acknowledge"
    exchanges = parse_gnb_control_exchanges([request, ack])
    assert exchanges[0]["t_ack_s"] - exchanges[0]["t_request_s"] == pytest.approx(0.002849, abs=2e-7)
    with pytest.raises(ValueError, match="second RIC Control Request"):
        parse_gnb_control_exchanges([request, request, ack])
    with pytest.raises(ValueError, match="without a preceding request"):
        parse_gnb_control_exchanges([ack])


def test_log_waiter_correlates_by_run_scoped_serial_order(tmp_path: Path) -> None:
    xapp_log = tmp_path / "xapp.jsonl"
    gnb_log = tmp_path / "gnb.log"
    xapp_log.write_text("")
    gnb_log.write_text("")
    waiter = LogCompletionWaiter(
        xapp_log, gnb_log, run_id="run-a", clock=iter([0.0, 0.1]).__next__, sleep=lambda _: None,
    )
    xapp_events = [
        {"event": "kpm", "t_utc": "2026-09-25T11:39:32.050000Z", "DRB.UEThpDl": 1000, "RRU.PrbTotDl": 10},
        {"event": "control_sent", "run_id": "other", "policy_id": "c3-other-00-x",
         "t_seen": "2026-09-25T11:39:32.059000Z", "t_control_sent": "2026-09-25T11:39:32.060000Z"},
        {"event": "control_sent", "run_id": "run-a", "policy_id": "c3-run-a-00-x",
         "t_seen": "2026-09-25T11:39:32.060000Z", "t_control_sent": "2026-09-25T11:39:32.062000Z"},
        {"event": "kpm", "t_utc": "2026-09-25T11:39:32.090000Z", "DRB.UEThpDl": 1200, "RRU.PrbTotDl": 10},
    ]
    xapp_log.write_text("\n".join(json.dumps(event) for event in xapp_events) + "\n")
    gnb_log.write_text(
        "2026-09-25T11:39:32.064483 [E2-DU   ] [I] Received RIC Control Request\n"
        "2026-09-25T11:39:32.067332 [E2-DU   ] [I] Sending E2 RIC Control Acknowledge\n"
    )
    completion = waiter.wait("c3-run-a-00-x", timeout_s=1.0)
    assert completion.t_gnb_ack_s - parse_timestamp("2026-09-25T11:39:32Z") == pytest.approx(0.067332)
    assert completion.kpm_change is not None and completion.kpm_change["DRB.UEThpDl"] == 1200


def test_policy_ids_are_unique_per_run_and_xapp_filters_other_runs() -> None:
    assert make_policy_id("run-a", 0, "i-1") == "c3-run-a-00-f105b11df4"
    assert make_policy_id("run-b", 0, "i-1") == "c3-run-b-00-f105b11df4"
    assert policy_belongs_to_run("c3-run-a-00-f105b11df4", "run-a")
    assert not policy_belongs_to_run("c3-run-b-00-f105b11df4", "run-a")


def test_docker_clock_probe_uses_lowest_response_delay_sample() -> None:
    class Result:
        def __init__(self, value: str) -> None:
            self.stdout = value

    times = iter([100.0, 102.0, 103.0, 104.0])
    results = iter([Result("101.0\n"), Result("103.8\n")])
    prober = DockerClockProber(
        "xapp", "gnb", samples=2, clock=times.__next__, runner=lambda *args, **kwargs: next(results),
    )
    point = prober._one("xapp")
    assert point == ClockPoint(host_reference_s=104.0, remote_s=103.8, round_trip_s=1.0)


def test_xapp_emits_control_only_after_successful_submission() -> None:
    class FakeRc:
        requestorID = 6

        def __init__(self, fail: bool) -> None:
            self.fail = fail

        def control_slice_level_prb_quota(self, *args, **kwargs) -> None:
            if self.fail:
                raise RuntimeError("send failed")

    control = control_for_policy({"action": "deprioritise", "class": "factory_control", "priority": "low"})
    events: list[dict] = []
    submit_control(FakeRc(False), "node", control, {"event": "control_sent"},
                   now=lambda: "2026-09-25T11:39:32.062000Z", emit_event=events.append)
    assert events == [{
        "event": "control_sent", "t_control_sent": "2026-09-25T11:39:32.062000Z", "e2_requestor_id": 7,
    }]
    with pytest.raises(RuntimeError, match="send failed"):
        submit_control(FakeRc(True), "node", control, {"event": "control_sent"}, emit_event=events.append)
    assert len(events) == 1


def test_first_kpm_change_uses_pre_control_baseline_and_material_threshold() -> None:
    events = [
        {"event": "kpm", "t_utc": "2026-09-25T11:39:32.051000Z", "DRB.UEThpDl": 28813, "RRU.PrbTotDl": 84},
        {"event": "control_sent", "t_control_sent": "2026-09-25T11:39:32.062000Z"},
        {"event": "kpm", "t_utc": "2026-09-25T11:39:32.274000Z", "DRB.UEThpDl": 28000, "RRU.PrbTotDl": 84},
        {"event": "kpm", "t_utc": "2026-09-25T11:39:32.690000Z", "DRB.UEThpDl": 20806, "RRU.PrbTotDl": 84},
    ]
    text = "\n".join(json.dumps(event) for event in events)
    change = first_kpm_change(text, parse_timestamp("2026-09-25T11:39:32.062000Z"))
    assert change is not None
    assert change["DRB.UEThpDl"] == 20806
    assert change["timestamp_s"] - parse_timestamp("2026-09-25T11:39:32.062000Z") == pytest.approx(0.628)


class FakeA1Server:
    """In-process A1 transport: it opens no socket and records every request."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, str, dict]] = []

    def request(self, method: str, url: str, body: bytes | None, timeout_s: float) -> HttpResult:
        decoded = json.loads(body) if body else {}
        self.requests.append((method, url, decoded))
        if method == "GET":
            return HttpResult(status=404, body=b"")
        return HttpResult(status=201, body=b'{"status":"CREATED"}')


def test_a1_client_times_fake_server_put_without_network() -> None:
    server = FakeA1Server()
    times = iter([100.0, 100.004])
    logs: list[dict] = []
    client = A1Client("http://fake-a1:8085", transport=server, clock=lambda: next(times), log=logs.append)
    assert client.ensure_policy_type() == 201
    assert server.requests[1][1].endswith("/policytype?id=20008")
    ack = client.put_policy("p-1", {"intent_id": "i-1", "ran_intent": {"priority": "low"}})
    assert ack.elapsed_s == pytest.approx(0.004)
    assert server.requests[-1][0] == "PUT"
    assert server.requests[-1][1].endswith("/A1-P/v2/policytypes/20008/policies/p-1")
    assert logs[0]["event"] == "a1_put_ack" and logs[0]["status"] == 201


class FakeInterpreter(Interpreter):
    def __init__(self, policy: dict[str, str]) -> None:
        super().__init__(name="Jev-1.13.0", deployment="hosted")
        self.policy = policy
        self.seen: list[str] = []

    def decide(self, case) -> Decision:
        self.seen.append(case.case_id)
        return Decision(labels=[self.policy], valid=True, latency_s=0.123, http_status=200)


class FakeWaiter:
    def wait(self, policy_id: str, *, timeout_s: float) -> Completion:
        assert policy_id == "c3-test-run-00-f105b11df4"
        return Completion(
            t_seen_s=154.45,
            t_control_sent_s=155.0,
            t_gnb_ack_s=156.0,
            t_first_kpm_change_s=157.2,
            kpm_change={"event": "kpm", "DRB.UEThpDl": 1000, "timestamp_s": 157.2},
        )


class FakeClockProber:
    def __init__(self) -> None:
        self.probes = iter([
            ClockProbeSet(
                xapp=ClockPoint(host_reference_s=90.0, remote_s=100.0, round_trip_s=0.002),
                gnb=ClockPoint(host_reference_s=90.0015, remote_s=100.0, round_trip_s=0.004),
            ),
            ClockProbeSet(
                xapp=ClockPoint(host_reference_s=190.0, remote_s=210.0, round_trip_s=0.003),
                gnb=ClockPoint(host_reference_s=190.001, remote_s=210.0, round_trip_s=0.005),
            ),
        ])

    def probe(self) -> ClockProbeSet:
        return next(self.probes)


def test_driver_writes_one_complete_jsonl_record(tmp_path: Path) -> None:
    truth = {
        "action": "deprioritise",
        "class": "factory_control",
        "scope": "stadium",
        "target_cluster": "none",
        "priority": "low",
        "prb_share": "unspecified",
        "edge_site": "unspecified",
        "latency_target": "unspecified",
        "duration": "unspecified",
    }
    pool = tmp_path / "pool.jsonl"
    pool.write_text(json.dumps({
        "intent_id": "i-1",
        "source_condition": "c21_fresh",
        "split": "test",
        "issuer": {"id": "operator", "kind": "operator"},
        "text": "Reduce factory control traffic in the stadium.",
        "actuated": {key: truth[key] for key in ("action", "class", "scope", "priority")},
        "truth": truth,
    }) + "\n")
    server = FakeA1Server()
    a1_times = iter([139.5, 139.6])
    a1 = A1Client("http://fake", transport=server, clock=lambda: next(a1_times))
    wall_times = iter([99.8, 100.0])
    out = tmp_path / "c3.jsonl"
    records = run_c3(
        interpreter=FakeInterpreter(truth),
        a1=a1,
        waiter=FakeWaiter(),
        pool_path=pool,
        out_path=out,
        iperf3_log_path=tmp_path / "iperf3.json",
        run_id="test-run",
        clock_prober=FakeClockProber(),
        limit=1,
        clock=lambda: next(wall_times),
    )
    assert len(records) == 1
    record = json.loads(out.read_text())
    assert record["ell_s"] == pytest.approx(0.123)  # adapter's monotonic timed operation, not 0.2 wall time
    assert record["a1_put_ack_latency_s"] == pytest.approx(0.1)
    assert record["delta_a1_s"] == pytest.approx(0.5)
    assert record["delta_e2_s"] == pytest.approx(1.0)
    assert record["kpm_change_upper_bound_s"] == pytest.approx(2.2)
    assert record["policy_id"] == "c3-test-run-00-f105b11df4"
    assert record["t_control_sent"] == "1970-01-01T00:02:20.000000Z"
    assert record["remote_timestamps"]["xapp_t_control_sent"] == "1970-01-01T00:02:35.000000Z"
    assert record["clock_probes"]["normalization"] == "linear_vm_to_host_via_xapp"
    shared_clock = record["clock_probes"]["shared_vm_clock"]
    assert shared_clock["before_offset_difference_s"] == pytest.approx(0.0015)
    assert shared_clock["after_offset_difference_s"] == pytest.approx(0.001)
    assert shared_clock["tolerance_s"] == pytest.approx(0.050)
    assert record["control"]["max_prb_ratio"] == 5
    assert record["iperf3_log_path"].endswith("iperf3.json")
    with pytest.raises(FileExistsError, match="refusing to append"):
        run_c3(
            interpreter=FakeInterpreter(truth), a1=a1, waiter=FakeWaiter(), pool_path=pool, out_path=out,
            iperf3_log_path="unused", run_id="test-run", clock_prober=FakeClockProber(), limit=1,
        )


def test_vm_internal_delta_ignores_independent_mapping_noise() -> None:
    clocks = ClockBracket(
        before=ClockProbeSet(
            xapp=ClockPoint(host_reference_s=100.0, remote_s=200.0, round_trip_s=0.001),
            gnb=ClockPoint(host_reference_s=99.9982, remote_s=200.0, round_trip_s=0.004),
        ),
        after=ClockProbeSet(
            xapp=ClockPoint(host_reference_s=110.0, remote_s=210.0, round_trip_s=0.001),
            gnb=ClockPoint(host_reference_s=109.9998, remote_s=210.0, round_trip_s=0.004),
        ),
    )
    evidence = clocks.verify_shared_vm_clock()
    sent_raw = 205.0
    ack_raw = 205.0004

    independently_mapped = clocks.gnb_to_host(ack_raw) - clocks.xapp_to_host(sent_raw)
    assert independently_mapped < 0
    assert ack_raw - sent_raw == pytest.approx(0.0004)
    assert evidence["max_abs_offset_difference_s"] == pytest.approx(0.0018)


def test_shared_vm_clock_disagreement_fails_closed() -> None:
    clocks = ClockBracket(
        before=ClockProbeSet(
            xapp=ClockPoint(host_reference_s=100.0, remote_s=200.0, round_trip_s=0.001),
            gnb=ClockPoint(host_reference_s=100.06, remote_s=200.0, round_trip_s=0.001),
        ),
        after=ClockProbeSet(
            xapp=ClockPoint(host_reference_s=110.0, remote_s=210.0, round_trip_s=0.001),
            gnb=ClockPoint(host_reference_s=110.06, remote_s=210.0, round_trip_s=0.001),
        ),
    )
    with pytest.raises(RuntimeError, match="shared VM clock probe disagreement"):
        clocks.verify_shared_vm_clock()


def test_probe_noise_below_gross_threshold_is_recorded_not_fatal() -> None:
    clocks = ClockBracket(
        before=ClockProbeSet(
            xapp=ClockPoint(host_reference_s=100.0, remote_s=200.0, round_trip_s=0.001),
            gnb=ClockPoint(host_reference_s=100.00215, remote_s=200.0, round_trip_s=0.001),
        ),
        after=ClockProbeSet(
            xapp=ClockPoint(host_reference_s=110.0, remote_s=210.0, round_trip_s=0.001),
            gnb=ClockPoint(host_reference_s=110.001, remote_s=210.0, round_trip_s=0.001),
        ),
    )
    evidence = clocks.verify_shared_vm_clock()
    assert evidence["max_abs_offset_difference_s"] == pytest.approx(0.00215)


def test_verify_same_kernel_requires_equal_boot_ids() -> None:
    from types import SimpleNamespace
    from src.ranbench.c3.driver import DockerClockProber

    def runner_for(ids):
        def run(cmd, **_kw):
            return SimpleNamespace(stdout=ids[cmd[2]] + "\n")
        return run

    same = DockerClockProber("xapp", "gnb", runner=runner_for({"xapp": "abc", "gnb": "abc"}))
    assert same.verify_same_kernel() == {"xapp_boot_id": "abc", "gnb_boot_id": "abc"}
    other = DockerClockProber("xapp", "gnb", runner=runner_for({"xapp": "abc", "gnb": "def"}))
    with pytest.raises(RuntimeError, match="do not share one kernel"):
        other.verify_same_kernel()
