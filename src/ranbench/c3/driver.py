"""Sequential live C3 driver for EXP-2026-003 RQ6."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import time
from typing import Any, Callable, Protocol
import uuid

from src.edgebench.interpreters.base import Decision, Interpreter
from src.ranbench.c3.a1 import A1Client, PutAcknowledgement, iso_utc
from src.ranbench.c3.mapping import control_for_policy
from src.ranbench.c3.parsers import (
    first_kpm_change,
    parse_gnb_control_exchanges,
    parse_timestamp,
    parse_xapp_controls,
)
from src.ranbench.interpreters import RanCase, build_interpreter

DEFAULT_POOL = Path(__file__).resolve().parents[3] / "data" / "ranbench" / "ranintent-v1" / "pool" / "c2c3_pool.jsonl"
DEFAULT_LIMIT = 30
DEFAULT_SEED = 20260925
# Both containers run on one Docker Desktop Linux VM kernel; that is verified structurally once per run (equal
# kernel boot_id). The per-intent probe difference is Docker-exec path noise (up to 2.1 ms observed) and is recorded;
# it only fails closed above a gross threshold that a different clock domain would exceed.
VM_CLOCK_OFFSET_TOLERANCE_S = 0.050
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


@dataclass(frozen=True)
class Completion:
    t_seen_s: float
    t_control_sent_s: float
    t_gnb_ack_s: float
    t_first_kpm_change_s: float | None
    kpm_change: dict[str, Any] | None


class CompletionWaiter(Protocol):
    def wait(self, policy_id: str, *, timeout_s: float) -> Completion: ...


@dataclass(frozen=True)
class ClockPoint:
    host_reference_s: float
    remote_s: float
    round_trip_s: float

    def to_dict(self) -> dict[str, float]:
        return {
            "host_reference_s": self.host_reference_s,
            "remote_s": self.remote_s,
            "round_trip_s": self.round_trip_s,
        }


@dataclass(frozen=True)
class ClockProbeSet:
    xapp: ClockPoint
    gnb: ClockPoint

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {"xapp": self.xapp.to_dict(), "gnb": self.gnb.to_dict()}


class ClockProber(Protocol):
    def probe(self) -> ClockProbeSet: ...


@dataclass(frozen=True)
class ClockBracket:
    before: ClockProbeSet
    after: ClockProbeSet

    @staticmethod
    def _to_host(remote_s: float, before: ClockPoint, after: ClockPoint) -> float:
        remote_span = after.remote_s - before.remote_s
        if remote_span <= 0:
            raise ValueError("remote clock probe did not advance")
        fraction = (remote_s - before.remote_s) / remote_span
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("remote event timestamp falls outside its before/after clock probes")
        return before.host_reference_s + fraction * (after.host_reference_s - before.host_reference_s)

    def xapp_to_host(self, remote_s: float) -> float:
        return self._to_host(remote_s, self.before.xapp, self.after.xapp)

    def gnb_to_host(self, remote_s: float) -> float:
        return self._to_host(remote_s, self.before.gnb, self.after.gnb)

    @staticmethod
    def _offset_difference(probes: ClockProbeSet) -> float:
        xapp_offset = probes.xapp.remote_s - probes.xapp.host_reference_s
        gnb_offset = probes.gnb.remote_s - probes.gnb.host_reference_s
        return xapp_offset - gnb_offset

    def verify_shared_vm_clock(
        self, *, tolerance_s: float = VM_CLOCK_OFFSET_TOLERANCE_S,
    ) -> dict[str, float]:
        differences = {
            "before_offset_difference_s": self._offset_difference(self.before),
            "after_offset_difference_s": self._offset_difference(self.after),
        }
        max_difference = max(abs(value) for value in differences.values())
        if max_difference > tolerance_s:
            raise RuntimeError(
                "xApp/gNB shared VM clock probe disagreement "
                f"{max_difference:.6f}s exceeds {tolerance_s:.6f}s tolerance; "
                "refusing corrupted latency evidence"
            )
        return {
            **differences,
            "max_abs_offset_difference_s": max_difference,
            "tolerance_s": tolerance_s,
        }

    def to_dict(self, *, shared_vm_clock: dict[str, float]) -> dict[str, Any]:
        return {
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "normalization": "linear_vm_to_host_via_xapp",
            "shared_vm_clock": shared_vm_clock,
        }


class DockerClockProber:
    """Estimate container clock mappings with midpoint-bounded Docker probes."""

    def __init__(
        self,
        xapp_container: str,
        gnb_container: str,
        *,
        clock: Callable[[], float] = time.time,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        samples: int = 5,
    ) -> None:
        self.xapp_container = xapp_container
        self.gnb_container = gnb_container
        self.clock = clock
        self.runner = runner
        self.samples = samples

    def _one(self, container: str) -> ClockPoint:
        points: list[ClockPoint] = []
        for _ in range(self.samples):
            started = self.clock()
            result = self.runner(
                ["docker", "exec", container, "date", "+%s.%N"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            finished = self.clock()
            try:
                remote = float(result.stdout.strip())
            except ValueError as exc:
                raise RuntimeError(f"invalid clock probe from Docker container {container!r}") from exc
            # `date` runs after Docker's request/setup latency and immediately
            # before the response.  The receive timestamp is therefore the
            # correct host-side boundary; a midpoint would inject half of the
            # strongly asymmetric Docker setup latency into every result.
            points.append(ClockPoint(finished, remote, finished - started))
        # The largest remote-host value has the smallest response-path delay,
        # analogous to selecting the minimum-delay NTP sample.
        return max(points, key=lambda point: point.remote_s - point.host_reference_s)

    def probe(self) -> ClockProbeSet:
        return ClockProbeSet(xapp=self._one(self.xapp_container), gnb=self._one(self.gnb_container))

    def verify_same_kernel(self) -> dict[str, str]:
        """Both containers must report the same kernel boot_id (one VM kernel, hence one clock)."""
        ids = {}
        for role, container in (("xapp", self.xapp_container), ("gnb", self.gnb_container)):
            result = self.runner(["docker", "exec", container, "cat", "/proc/sys/kernel/random/boot_id"],
                                 check=True, capture_output=True, text=True, timeout=5.0)
            ids[f"{role}_boot_id"] = result.stdout.strip()
        if not ids["xapp_boot_id"] or ids["xapp_boot_id"] != ids["gnb_boot_id"]:
            raise RuntimeError(f"xApp and gNB containers do not share one kernel: {ids}; refusing latency evidence")
        return ids


class LogCompletionWaiter:
    """Correlate run-scoped xApp controls with strictly serial gNB exchanges."""

    def __init__(
        self,
        xapp_log: str | Path,
        gnb_log: str | Path,
        *,
        run_id: str,
        poll_s: float = 0.02,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.xapp_log = Path(xapp_log)
        self.gnb_log = Path(gnb_log)
        self.run_id = run_id
        self.poll_s = poll_s
        self.clock = clock
        self.sleep = sleep
        self._gnb_start_line = len(self._read(self.gnb_log).splitlines())
        self._completed = 0

    @staticmethod
    def _read(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            return ""

    def wait(self, policy_id: str, *, timeout_s: float) -> Completion:
        deadline = self.clock() + timeout_s
        control: dict[str, Any] | None = None
        ack: float | None = None
        kpm: dict[str, Any] | None = None
        while self.clock() < deadline:
            xapp_text = self._read(self.xapp_log)
            controls = [
                event for event in parse_xapp_controls(xapp_text)
                if event.get("run_id") == self.run_id
            ]
            if len({str(event.get("policy_id")) for event in controls}) != len(controls):
                raise RuntimeError(f"duplicate control_sent event in C3 run {self.run_id}")
            if len(controls) > self._completed:
                next_policy = str(controls[self._completed].get("policy_id"))
                if next_policy != policy_id:
                    raise RuntimeError(
                        f"unexpected next xApp control {next_policy!r}; waiting for {policy_id!r}"
                    )
                control = controls[self._completed]
            if control is not None:
                sent = parse_timestamp(str(control["t_control_sent"]))
                gnb_lines = self._read(self.gnb_log).splitlines()[self._gnb_start_line:]
                exchanges = parse_gnb_control_exchanges(gnb_lines)
                if len(exchanges) > len(controls):
                    raise RuntimeError("gNB control exchange has no run-scoped xApp control")
                if len(exchanges) > self._completed:
                    ack = float(exchanges[self._completed]["t_ack_s"])
                kpm = first_kpm_change(xapp_text, after_s=sent)
                if ack is not None and kpm is not None:
                    completion = Completion(
                        t_seen_s=parse_timestamp(str(control["t_seen"])),
                        t_control_sent_s=sent,
                        t_gnb_ack_s=ack,
                        t_first_kpm_change_s=float(kpm["timestamp_s"]),
                        kpm_change=kpm,
                    )
                    self._completed += 1
                    return completion
            self.sleep(self.poll_s)
        if control is None:
            raise TimeoutError(f"xApp did not log control_sent for {policy_id} within {timeout_s:g}s")
        sent = parse_timestamp(str(control["t_control_sent"]))
        if ack is None:
            raise TimeoutError(f"gNB did not acknowledge {policy_id} within {timeout_s:g}s")
        # A no-op quota transition can have no KPM change.  ACK is the C3 integrity gate;
        # preserve a null upper-bound observation rather than inventing a timestamp.
        completion = Completion(
            t_seen_s=parse_timestamp(str(control["t_seen"])),
            t_control_sent_s=sent,
            t_gnb_ack_s=ack,
            t_first_kpm_change_s=None,
            kpm_change=None,
        )
        self._completed += 1
        return completion


def load_pool(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    clean = [row for row in rows if row.get("actuated") and row.get("truth") and row.get("intent_id")]
    if len(clean) != len(rows):
        raise ValueError(f"{path} contains a row that is not a clean actuatable intent")
    return clean


def select_intents(rows: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if limit < 1 or limit > len(rows):
        raise ValueError(f"limit must be between 1 and {len(rows)}, got {limit}")
    selected = list(rows)
    random.Random(seed).shuffle(selected)
    return selected[:limit]


def ran_case(row: dict[str, Any]) -> RanCase:
    return RanCase(
        case_id=str(row["intent_id"]),
        condition=str(row.get("source_condition", "c21_fresh")),
        split=str(row.get("split", "test")),
        half="named_scope",
        issuer=dict(row["issuer"]),
        text=str(row["text"]),
        telemetry="(not required: this is a named-scope C3 intent)",
        truth=dict(row["truth"]),
        meta={"actuated": row["actuated"], "c2_class": row.get("c2_class")},
    )


def _policy(decision: Decision) -> dict[str, str] | None:
    if not decision.valid or not decision.labels or not isinstance(decision.labels[0], dict):
        return None
    return dict(decision.labels[0])


def make_policy_id(run_id: str, sequence: int, case_id: str) -> str:
    case_token = hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:10]
    return f"c3-{run_id}-{sequence:02d}-{case_token}"


def run_c3(
    *,
    interpreter: Interpreter,
    a1: A1Client,
    waiter: CompletionWaiter,
    pool_path: str | Path,
    out_path: str | Path,
    iperf3_log_path: str | Path,
    run_id: str,
    clock_prober: ClockProber,
    limit: int = DEFAULT_LIMIT,
    seed: int = DEFAULT_SEED,
    completion_timeout_s: float = 8.0,
    clock: Callable[[], float] = time.time,
) -> list[dict[str, Any]]:
    """Interpret and actuate one intent at a time, appending one durable JSONL row per intent."""
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("run_id must be 1-64 ASCII letters, digits, dots, underscores, or hyphens")
    out = Path(out_path)
    if out.exists() and out.stat().st_size:
        raise FileExistsError(f"refusing to append to non-empty C3 output {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = select_intents(load_pool(pool_path), limit, seed)
    pool_sha256 = hashlib.sha256(Path(pool_path).read_bytes()).hexdigest()
    written: list[dict[str, Any]] = []
    same_kernel_check = getattr(clock_prober, "verify_same_kernel", None)
    same_kernel = same_kernel_check() if same_kernel_check is not None else None
    with out.open("x", encoding="utf-8") as handle:
        for sequence, source in enumerate(rows):
            case = ran_case(source)
            t_issue = clock()
            decision = interpreter.decide(case)
            t_decision = clock()
            policy = _policy(decision)
            record: dict[str, Any] = {
                "experiment": "EXP-2026-003",
                "component": "C3",
                "rq": "RQ6",
                "sequence": sequence,
                "run_id": run_id,
                "seed": seed,
                "pool_sha256": pool_sha256,
                "intent_id": case.case_id,
                "interpreter": interpreter.name,
                "policy": policy,
                "decision_valid": decision.valid,
                "decision_error": decision.error_type,
                "ell_s": decision.latency_s,
                "adapter_latency_s": decision.latency_s,
                "t_intent_issued": iso_utc(t_issue),
                "t_decision_returned": iso_utc(t_decision),
                "iperf3_log_path": str(iperf3_log_path),
                "policy_id": None,
                "control": None,
                "actuation_error": None,
                "a1_http_status": None,
                "a1_put_ack_latency_s": None,
                "delta_a1_s": None,
                "a1_poll_wait_s": None,
                "delta_e2_s": None,
                "kpm_change_upper_bound_s": None,
                "kpm_change": None,
                "t_a1_put_requested": None,
                "t_a1_put_acknowledged": None,
                "t_seen": None,
                "t_control_sent": None,
                "t_gnb_acknowledged": None,
                "t_first_kpm_change": None,
                "remote_timestamps": None,
                "clock_probes": None,
            }
            if policy is not None:
                try:
                    control = control_for_policy(policy)
                except ValueError as exc:
                    record["actuation_error"] = str(exc)
                    handle.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    written.append(record)
                    continue
                policy_id = make_policy_id(run_id, sequence, case.case_id)
                probes_before = clock_prober.probe()
                ack: PutAcknowledgement = a1.put_policy(
                    policy_id,
                    {
                        "intent_id": case.case_id,
                        "interpreter": interpreter.name,
                        "ran_intent": policy,
                        "expected_control": control.to_dict(),
                    },
                )
                completion = waiter.wait(policy_id, timeout_s=completion_timeout_s)
                probes_after = clock_prober.probe()
                clocks = ClockBracket(probes_before, probes_after)
                shared_vm_clock = clocks.verify_shared_vm_clock()
                seen_host = clocks.xapp_to_host(completion.t_seen_s)
                sent_host = clocks.xapp_to_host(completion.t_control_sent_s)
                # The xApp and gNB share the Docker Desktop Linux VM clock.
                # Use one mapping only to retain host-normalized evidence; VM-internal
                # latency intervals must stay in the raw clock and must not subtract
                # independently noisy remote-to-host mappings.
                ack_host = clocks.xapp_to_host(completion.t_gnb_ack_s)
                kpm_host = (
                    clocks.xapp_to_host(completion.t_first_kpm_change_s)
                    if completion.t_first_kpm_change_s is not None else None
                )
                delta_a1 = sent_host - ack.t_request_s
                poll_wait = sent_host - ack.t_ack_s
                delta_e2 = completion.t_gnb_ack_s - completion.t_control_sent_s
                kpm_upper = (
                    completion.t_first_kpm_change_s - completion.t_control_sent_s
                    if completion.t_first_kpm_change_s is not None else None
                )
                if (
                    delta_a1 < 0
                    or poll_wait < 0
                    or delta_e2 < 0
                    or (kpm_upper is not None and kpm_upper < 0)
                ):
                    raise RuntimeError(
                        "C3 measurement produced a negative interval; refusing corrupted latency evidence"
                    )
                record.update({
                    "policy_id": policy_id,
                    "control": control.to_dict(),
                    "a1_http_status": ack.status,
                    "a1_put_ack_latency_s": ack.elapsed_s,
                    "delta_a1_s": delta_a1,
                    "a1_poll_wait_s": poll_wait,
                    "delta_e2_s": delta_e2,
                    "kpm_change_upper_bound_s": kpm_upper,
                    "kpm_change": completion.kpm_change,
                    "shared_kernel": same_kernel,
                    "t_a1_put_requested": iso_utc(ack.t_request_s),
                    "t_a1_put_acknowledged": iso_utc(ack.t_ack_s),
                    "t_seen": iso_utc(seen_host),
                    "t_control_sent": iso_utc(sent_host),
                    "t_gnb_acknowledged": iso_utc(ack_host),
                    "t_first_kpm_change": (
                        iso_utc(kpm_host) if kpm_host is not None else None
                    ),
                    "remote_timestamps": {
                        "xapp_t_seen": iso_utc(completion.t_seen_s),
                        "xapp_t_control_sent": iso_utc(completion.t_control_sent_s),
                        "gnb_t_acknowledged": iso_utc(completion.t_gnb_ack_s),
                        "xapp_t_first_kpm_change": (
                            iso_utc(completion.t_first_kpm_change_s)
                            if completion.t_first_kpm_change_s is not None else None
                        ),
                    },
                    "clock_probes": clocks.to_dict(shared_vm_clock=shared_vm_clock),
                })
            handle.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            written.append(record)
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ranbench.c3", description=__doc__)
    parser.add_argument("--interpreter", required=True)
    parser.add_argument("--pool", default=str(DEFAULT_POOL))
    parser.add_argument("--out", required=True)
    parser.add_argument("--iperf3-log", required=True)
    parser.add_argument("--xapp-log", required=True)
    parser.add_argument("--gnb-log", required=True)
    parser.add_argument("--a1-url", default="http://127.0.0.1:8085")
    parser.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY"))
    parser.add_argument("--base-url")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--completion-timeout", type=float, default=8.0)
    parser.add_argument("--run-id", default=os.environ.get("C3_RUN_ID") or uuid.uuid4().hex[:12])
    parser.add_argument("--xapp-container", default="python_xapp_runner")
    parser.add_argument("--gnb-container", default="ocudu_gnb")
    args = parser.parse_args(argv)

    interpreter = build_interpreter(args.interpreter, api_key=args.api_key, base_url=args.base_url)
    a1_log_path = Path(args.out).with_suffix(".a1.jsonl")
    a1_log_path.parent.mkdir(parents=True, exist_ok=True)
    with a1_log_path.open("x", encoding="utf-8") as a1_log:
        def log_a1(event: dict[str, Any]) -> None:
            a1_log.write(json.dumps(event, separators=(",", ":"), sort_keys=True) + "\n")
            a1_log.flush()

        a1 = A1Client(args.a1_url, log=log_a1)
        a1.ensure_policy_type()
        records = run_c3(
            interpreter=interpreter,
            a1=a1,
            waiter=LogCompletionWaiter(args.xapp_log, args.gnb_log, run_id=args.run_id),
            pool_path=args.pool,
            out_path=args.out,
            iperf3_log_path=args.iperf3_log,
            run_id=args.run_id,
            clock_prober=DockerClockProber(args.xapp_container, args.gnb_container),
            limit=args.limit,
            seed=args.seed,
            completion_timeout_s=args.completion_timeout,
        )
    print(f"C3 wrote {len(records)} records for {interpreter.name} to {args.out}")


if __name__ == "__main__":
    main()
