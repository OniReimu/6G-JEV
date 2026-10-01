"""Live RQ3 interpreter load traces for EXP-2026-003.

One run sends 300 clean, actuatable pool intents through four live interpretation slots. Arrivals are a seeded
Poisson process, admission is an unbounded FIFO queue, and a slot remains occupied until ``decide`` returns. The
trace contains the queue timeline and is directly consumable by ``ranbench.c2.streams.stream_from_trace``; the
separate ledger keeps the existing Edgebench row schema.

Runs are deliberately not resumable. A partial wall-clock run has already consumed queue and (for AnyJev) online
state, so replaying only its missing suffix would not reproduce the registered process. Any non-empty output
directory is refused and a new run tag/directory is required.

CLI example::

    python -m src.ranbench.rq3_load run --interpreter Jev-1.13.0 --rate 1 \
        --out runs/EXP-2026-003/RQ3/Jev-1.13.0/rate_1 --client-location "Sydney, Australia"
"""
from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import heapq
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np

from src.credentials import load_openrouter_key
from src.edgebench.interpreters.base import Decision, Interpreter
from src.edgebench.interpreters.transport import (
    get_post_timeout_healthy,
    get_post_timeout_wait,
    get_send_marker,
    reset_send_marker,
)
from src.edgebench.ledger import LedgerWriter, resolve_git_provenance
from src.ranbench.c2.queueing import utilisation
from src.ranbench.corpus.spec import load_config
from src.ranbench.interpreters.anyjev_l0 import LABEL_FALLBACK_ERROR
from src.ranbench.interpreters.common import RanCase, score_policy
from src.ranbench.interpreters.manifest import build_interpreter, load_manifest
from src.ranbench.interpreters.runner import (
    ERROR_RATE_RERUN_THRESHOLD,
    SERVER_DOWN_ERRORS,
    SERVER_ERROR_STREAK_STOP,
    STOP_ERRORS,
    _is_transport_error,
    verify_corpus_file,
)

RQ = "RQ3"
DEFAULT_SEED = 20260925
DEFAULT_INTENTS = 300
DEFAULT_SLOTS = 4
DEFAULT_CORPUS_DIR = Path("data/ranbench/ranintent-v1")
POOL_REL = "pool/c2c3_pool.jsonl"
TRACE_FILE = "trace.jsonl"
LEDGER_FILE = "ledger.jsonl"
INTEGRITY_FILE = "integrity.json"
ORDER_FILE = "anyjev_order.jsonl"


def _rate_tag(rate: float) -> str:
    return f"rate_{rate:g}".replace(".", "p")


def load_verified_pool(
    corpus_dir: str | Path, allow_stub: bool = False
) -> tuple[list[dict[str, Any]], str, str]:
    """Load the frozen C2/C3 pool after checking its corpus-manifest hash."""
    root = Path(corpus_dir)
    pool_sha256, manifest_sha256 = verify_corpus_file(root, POOL_REL, allow_stub=allow_stub)
    rows = [json.loads(line) for line in (root / POOL_REL).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("Refusing to start: the C2/C3 pool is empty")
    ids = [str(r.get("intent_id", "")) for r in rows]
    if any(not i for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("Refusing to start: pool intent_id values must be non-empty and unique")
    required = {"source_condition", "split", "issuer", "text", "truth", "actuated", "c2_class"}
    missing = [(r["intent_id"], sorted(required - set(r))) for r in rows if required - set(r)]
    if missing:
        raise ValueError(f"Refusing to start: malformed pool row {missing[0][0]} missing {missing[0][1]}")
    return rows, pool_sha256, manifest_sha256


def make_load_schedule(
    pool: Sequence[dict[str, Any]],
    rate: float,
    n_intents: int = DEFAULT_INTENTS,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    """Seeded Poisson arrivals plus without-replacement pool passes.

    Reusing the seed at another rate reuses both the underlying exponential variates and intent order; only the
    inter-arrival scale changes. A fresh pool permutation is drawn after every complete pass.
    """
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("rate must be finite and positive")
    if n_intents < 1:
        raise ValueError("n_intents must be positive")
    if not pool:
        raise ValueError("pool must not be empty")
    rng = np.random.default_rng(seed)
    arrivals = np.cumsum(rng.exponential(1.0 / rate, size=n_intents))
    order: list[int] = []
    while len(order) < n_intents:
        order.extend(int(i) for i in rng.permutation(len(pool)))
    schedule: list[dict[str, Any]] = []
    for index, (arrival, pool_index) in enumerate(zip(arrivals, order[:n_intents])):
        record = pool[pool_index]
        schedule.append(
            {
                "event_index": index,
                "event_id": f"rq3_{index:04d}",
                "intent_id": str(record["intent_id"]),
                "arrival_s": float(arrival),
                "pool_index": pool_index,
                "pool_pass": index // len(pool),
            }
        )
    return schedule


def fifo_timings(
    arrivals_s: Sequence[float], service_s: Sequence[float], slots: int = DEFAULT_SLOTS
) -> list[dict[str, float | int]]:
    """Hand-checkable FIFO timeline with the same slot semantics as the live runner and edgebench/e2e."""
    if not isinstance(slots, int) or slots < 1:
        raise ValueError("slots must be a positive integer")
    if len(arrivals_s) != len(service_s):
        raise ValueError("arrivals_s and service_s must have the same length")
    previous = -math.inf
    free: list[tuple[float, int]] = [(0.0, slot) for slot in range(slots)]
    heapq.heapify(free)
    rows: list[dict[str, float | int]] = []
    for arrival, service in zip(arrivals_s, service_s):
        arrival, service = float(arrival), float(service)
        if not math.isfinite(arrival) or arrival < previous:
            raise ValueError("arrivals_s must be finite and non-decreasing")
        if not math.isfinite(service) or service < 0:
            raise ValueError("service_s must be finite and non-negative")
        available, slot = heapq.heappop(free)
        start = max(arrival, available)
        completion = start + service
        heapq.heappush(free, (completion, slot))
        rows.append(
            {
                "slot": slot,
                "arrival_s": arrival,
                "service_start_s": start,
                "queue_wait_s": start - arrival,
                "service_latency_s": service,
                "completion_s": completion,
            }
        )
        previous = arrival
    return rows


def _case(pool_row: dict[str, Any], case_id: str | None = None) -> RanCase:
    # Pool intents are the named-scope half and therefore do not depend on telemetry. An empty table preserves the
    # same reading-rules/case rendering used by C1 without inventing state.
    return RanCase(
        case_id=case_id or str(pool_row["intent_id"]),
        condition=str(pool_row["source_condition"]),
        split=str(pool_row["split"]),
        half="named_scope",
        issuer=dict(pool_row["issuer"]),
        text=str(pool_row["text"]),
        telemetry="",
        truth=dict(pool_row["truth"]),
        meta={"rq3_pool": True},
    )


def _call(interpreter: Interpreter, case: RanCase) -> tuple[Decision, bool]:
    """Call one adapter in its slot thread and preserve the interpreter runner's exception/timing vocabulary."""
    reset_send_marker()
    try:
        decision = interpreter.decide(case)
    except Exception as exc:
        sent = get_send_marker()
        decision = Decision(
            labels=[],
            valid=False,
            error_type=type(exc).__name__,
            latency_s=(time.perf_counter() - sent[1]) if sent else None,
            t_send_wall=sent[0] if sent else 0.0,
            t_recv_wall=time.time() if sent else 0.0,
            provider=getattr(interpreter, "provider_slug", "") or "",
        )
    decision.post_timeout_wait_s = get_post_timeout_wait()
    return decision, get_post_timeout_healthy() is False


def _stop_reason(
    interpreter: Interpreter,
    decision: Decision,
    unhealthy: bool,
    spend: float,
    cap: float,
    server_error_streak: int,
) -> str | None:
    if spend >= cap:
        return f"spend_cap_exceeded (${spend:.4f} >= ${cap})"
    if decision.http_status in (401, 403):
        return f"HTTP_{decision.http_status}"
    if decision.error_type in STOP_ERRORS:
        return str(decision.error_type)
    if interpreter.deployment == "self-hosted":
        if unhealthy or (decision.http_status is None and decision.error_type in SERVER_DOWN_ERRORS):
            return "self_hosted_server_down"
        if server_error_streak >= SERVER_ERROR_STREAK_STOP:
            return "self_hosted_server_error"
    return None


def _prepare_output(path: str | Path) -> Path:
    out = Path(path)
    if out.exists() and not out.is_dir():
        raise RuntimeError(f"Refusing to start: output path is not a directory: {out}")
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(
            f"Refusing to resume or overwrite non-empty output directory {out}; use a new run tag/directory"
        )
    out.mkdir(parents=True, exist_ok=True)
    return out


async def _run_live(
    schedule: list[dict[str, Any]],
    pool: Sequence[dict[str, Any]],
    interpreters: Sequence[Interpreter],
    writer: LedgerWriter,
    trace_path: Path,
    cfg: dict[str, Any],
    spend_cap_usd: float,
) -> dict[str, Any]:
    slots = len(interpreters)
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()  # unbounded FIFO
    stop = asyncio.Event()
    state: dict[str, Any] = {
        "stop_reason": None,
        "spend": 0.0,
        "cost_unknown_rows": 0,
        "completed": 0,
        "aborted": 0,
        "valid": 0,
        "services": [],
        "errors": {},
        "transport_errors": 0,
        "server_error_streak": 0,
        "per_slot": [0] * slots,
    }
    monotonic_start = time.monotonic()
    deferred_rows: list[dict[str, Any]] = []

    def now() -> float:
        return time.monotonic() - monotonic_start

    def write_trace(stream: Any, row: dict[str, Any]) -> None:
        stream.write(json.dumps(row) + "\n")
        stream.flush()

    async def producer() -> None:
        for event in schedule:
            delay = event["arrival_s"] - now()
            if delay > 0:
                try:
                    await asyncio.wait_for(stop.wait(), timeout=delay)
                except asyncio.TimeoutError:
                    pass
            if stop.is_set():
                break
            queued = dict(event)
            queued["observed_arrival_s"] = now()
            await queue.put(queued)
        for _ in range(slots):
            await queue.put(None)

    async def worker(slot: int, interpreter: Interpreter, stream: Any, executor: ThreadPoolExecutor) -> None:
        while True:
            event = await queue.get()
            if event is None:
                queue.task_done()
                return
            pool_row = pool[int(event["pool_index"])]
            if stop.is_set():
                terminal = now()
                write_trace(
                    stream,
                    {
                        **event,
                        "run_id": writer.run_id,
                        "model": interpreter.name,
                        "slot": slot,
                        "status": "aborted_before_dispatch",
                        "service_start_s": None,
                        "queue_wait_s": None,
                        "service_latency_s": None,
                        "completion_s": terminal,
                        "policy": None,
                        "labels": None,
                        "valid": False,
                        "error_type": state["stop_reason"],
                    },
                )
                state["aborted"] += 1
                queue.task_done()
                continue

            begin = now()
            case = _case(pool_row)
            decision, unhealthy = await loop.run_in_executor(executor, _call, interpreter, case)
            end = now()
            service = end - begin
            scores = score_policy(decision, case, cfg)
            row = dict(
                model=interpreter.name,
                case_id=case.case_id,
                repeat=int(event["pool_pass"]),
                decision=decision,
                platform=getattr(interpreter, "platform", None),
                thread=f"slot-{slot}",
                **scores,
            )
            if getattr(decision, "count_input_tokens", None) is not None:
                deferred_rows.append(row)  # ledger token count after the cell, off the slot
            else:
                writer.write_row(**row)
            policy = decision.labels[0] if decision.labels and isinstance(decision.labels[0], dict) else None
            write_trace(
                stream,
                {
                    **event,
                    "run_id": writer.run_id,
                    "model": interpreter.name,
                    "case_id": case.case_id,
                    "repeat": int(event["pool_pass"]),
                    "slot": slot,
                    "status": "decision_recorded",
                    "arrival_lag_s": event["observed_arrival_s"] - event["arrival_s"],
                    "service_start_s": begin,
                    "queue_wait_s": begin - event["arrival_s"],
                    "service_latency_s": service,
                    "latency_s": service,
                    "adapter_latency_s": decision.latency_s,
                    "completion_s": end,
                    "policy": policy,
                    "labels": policy,
                    "valid": bool(decision.valid),
                    "error_type": decision.error_type,
                    "http_status": decision.http_status,
                    "resolved_model": decision.resolved_model,
                    "provider": decision.provider,
                    "cost_usd": decision.cost_usd,
                },
            )
            state["completed"] += 1
            state["valid"] += int(bool(decision.valid))
            state["services"].append(service)
            state["per_slot"][slot] += 1
            if decision.error_type:
                errors = state["errors"]
                errors[decision.error_type] = errors.get(decision.error_type, 0) + 1
            if _is_transport_error({"http_status": decision.http_status}):
                state["transport_errors"] += 1
            is_5xx = decision.http_status is not None and 500 <= decision.http_status < 600
            state["server_error_streak"] = state["server_error_streak"] + 1 if is_5xx else 0
            if decision.cost_usd is None:
                state["cost_unknown_rows"] += 1
            else:
                state["spend"] += float(decision.cost_usd)
            reason = _stop_reason(
                interpreter,
                decision,
                unhealthy,
                state["spend"],
                spend_cap_usd,
                state["server_error_streak"],
            )
            if reason and not state["stop_reason"]:
                state["stop_reason"] = reason
                stop.set()
            queue.task_done()

    with open(trace_path, "w", encoding="utf-8") as stream, ThreadPoolExecutor(max_workers=slots) as executor:
        workers = [asyncio.create_task(worker(i, interpreter, stream, executor))
                   for i, interpreter in enumerate(interpreters)]
        await producer()
        await asyncio.gather(*workers)
    state["elapsed_s"] = now()
    for row in deferred_rows:
        try:
            row["decision"].input_tokens = row["decision"].count_input_tokens()
        except Exception:  # noqa: BLE001 - a bookkeeping failure must not drop the row; the count stays None
            row["decision"].input_tokens = None
        writer.write_row(**row)
    return state


def run_rq3_load(
    corpus_dir: str | Path,
    interpreter_name: str,
    rate: float,
    out_dir: str | Path,
    *,
    slots: int = DEFAULT_SLOTS,
    n_intents: int = DEFAULT_INTENTS,
    seed: int = DEFAULT_SEED,
    spend_cap_usd: float = 20.0,
    client_location: str | None = None,
    manifest_path: str | Path | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    allow_dirty: bool = False,
    allow_stub: bool = False,
    warmup: int | None = None,
    interpreters: Sequence[Interpreter] | None = None,
) -> dict[str, Any]:
    """Run one interpreter/rate cell and return the written integrity record."""
    if not isinstance(slots, int) or slots < 1:
        raise ValueError("slots must be a positive integer")
    if not math.isfinite(spend_cap_usd) or spend_cap_usd <= 0:
        raise ValueError("spend_cap_usd must be finite and positive")
    out_path = Path(out_dir)
    if out_path.exists() and (not out_path.is_dir() or any(out_path.iterdir())):
        # Refuse before provenance checks, adapter construction, or any possible live call.
        raise RuntimeError(
            f"Refusing to resume or overwrite non-empty output directory {out_path}; use a new run tag/directory"
        )

    git_sha, is_dirty, git_source, provenance_tree_sha256 = resolve_git_provenance()
    if is_dirty and not allow_dirty:
        raise RuntimeError(
            "Refusing to start: git working tree has modified tracked files. Pass allow_dirty to proceed."
        )
    pool, pool_sha256, corpus_manifest_sha256 = load_verified_pool(corpus_dir, allow_stub=allow_stub)
    schedule = make_load_schedule(pool, rate, n_intents=n_intents, seed=seed)
    manifest = load_manifest(manifest_path)
    if interpreter_name not in manifest and interpreters is None:
        raise KeyError(f"Interpreter '{interpreter_name}' not in manifest; available: {sorted(manifest)}")

    if interpreters is None:
        slot_clients = [
            build_interpreter(interpreter_name, manifest, api_key=api_key, base_url=base_url) for _ in range(slots)
        ]
    else:
        slot_clients = list(interpreters)
    if len(slot_clients) != slots:
        raise ValueError(f"expected one interpreter client per slot ({slots}), got {len(slot_clients)}")
    for client in slot_clients:
        if hasattr(client, "defer_input_tokens"):
            client.defer_input_tokens = True  # AnyJev: the ledger token count must not occupy a slot
    if len({id(interpreter) for interpreter in slot_clients}) != slots:
        raise ValueError("each interpretation slot needs a distinct interpreter client")
    if {interpreter.name for interpreter in slot_clients} != {interpreter_name}:
        raise ValueError("every slot interpreter must have interpreter_name")
    deployments = {interpreter.deployment for interpreter in slot_clients}
    if len(deployments) != 1:
        raise ValueError("all slot interpreters must use the same deployment")
    deployment = next(iter(deployments))
    if deployment == "hosted" and not client_location:
        raise ValueError("Hosted runs record the client location: pass client_location")

    out = _prepare_output(out_path)
    condition = _rate_tag(rate)
    run_id = f"{RQ}_{interpreter_name}_{condition}_{seed}"
    cases_sha256 = pool_sha256
    writer = LedgerWriter(
        path=out / LEDGER_FILE,
        run_id=run_id,
        git_sha=git_sha,
        rq=RQ,
        condition=condition,
        allow_dirty=allow_dirty,
        git_source=git_source,
        provenance_tree_sha256=provenance_tree_sha256,
        cases_sha256=cases_sha256,
    )
    cfg = load_config(Path(corpus_dir) / "config.json")
    if warmup is None:
        warmup_n = 2 if any(getattr(i, "order_dependent", False) for i in slot_clients) else 0
    else:
        warmup_n = warmup
    if warmup_n < 0:
        writer.close()
        raise ValueError("warmup must be non-negative")
    warmup_outcomes: list[dict[str, Any]] = []
    stop_reason: str | None = None
    for slot, interpreter in enumerate(slot_clients):
        for index in range(warmup_n):
            pool_row = pool[int(schedule[index % len(schedule)]["pool_index"])]
            case = _case(pool_row, case_id=f"warmup_slot{slot}_{index}_{pool_row['intent_id']}")
            if hasattr(interpreter, "warmup"):
                entry = interpreter.warmup(case)  # type: ignore[attr-defined]
                error = entry.get("error")
            else:
                decision, _ = _call(interpreter, case)
                failed = decision.http_status != 200 or decision.error_type == LABEL_FALLBACK_ERROR
                error = decision.error_type if failed else None
            warmup_outcomes.append({"slot": slot, "case_id": case.case_id, "error": error})
            if error and stop_reason is None:
                stop_reason = f"warmup_failed (slot={slot},case={case.case_id},error={error})"

    started_at = datetime.now(timezone.utc).isoformat()
    if stop_reason is None:
        state = asyncio.run(
            _run_live(
                schedule=schedule,
                pool=pool,
                interpreters=slot_clients,
                writer=writer,
                trace_path=out / TRACE_FILE,
                cfg=cfg,
                spend_cap_usd=spend_cap_usd,
            )
        )
        stop_reason = state["stop_reason"]
    else:
        (out / TRACE_FILE).write_text("", encoding="utf-8")
        state = {
            "stop_reason": stop_reason,
            "spend": 0.0,
            "cost_unknown_rows": 0,
            "completed": 0,
            "aborted": 0,
            "valid": 0,
            "services": [],
            "errors": {},
            "transport_errors": 0,
            "server_error_streak": 0,
            "per_slot": [0] * slots,
            "elapsed_s": 0.0,
        }
    writer.close()

    order_rows = 0
    with open(out / ORDER_FILE, "w", encoding="utf-8") as order_stream:
        for slot, interpreter in enumerate(slot_clients):
            for entry in getattr(interpreter, "history", []):
                row = {"run_id": run_id, "model": interpreter.name, "slot": slot, **entry}
                order_stream.write(json.dumps(row) + "\n")
                order_rows += 1
    if order_rows == 0:
        (out / ORDER_FILE).unlink()

    services = np.asarray(state["services"], dtype=float)
    util = utilisation(rate, services, slots) if services.size else {"rho": None, "non_stationary": None}
    transport_error_rate = (
        state["transport_errors"] / state["completed"] if state["completed"] else None
    )
    rerun_block_required = bool(state["completed"]) and (
        transport_error_rate is not None and transport_error_rate > ERROR_RATE_RERUN_THRESHOLD
    )
    degradation_reason = None
    if rerun_block_required:
        degradation_reason = (
            "transport_error_rate_exceeded "
            f"({state['transport_errors']}/{state['completed']}={transport_error_rate:.6f} "
            f"> {ERROR_RATE_RERUN_THRESHOLD:.6f})"
        )
        if stop_reason is None:
            stop_reason = degradation_reason
    schedule_sha256 = hashlib.sha256(
        json.dumps(schedule, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    report = {
        "run_id": run_id,
        "rq": RQ,
        "model": interpreter_name,
        "deployment": deployment,
        "rate_per_s": rate,
        "slots": slots,
        "seed": seed,
        "expected_intents": n_intents,
        "completed_intents": state["completed"],
        "aborted_after_arrival": state["aborted"],
        "valid_intents": state["valid"],
        "complete": stop_reason is None and state["completed"] == n_intents,
        "stop_reason": stop_reason,
        "degradation_reason": degradation_reason,
        "transport_error_rows": state["transport_errors"],
        "transport_error_rate": transport_error_rate,
        "transport_error_threshold": ERROR_RATE_RERUN_THRESHOLD,
        "rerun_block_required": rerun_block_required,
        "rho": util["rho"],
        "non_stationary": util["non_stationary"],
        "mean_service_latency_s": float(np.mean(services)) if services.size else None,
        "elapsed_wall_s": state["elapsed_s"],
        "total_spend_usd": state["spend"],
        "spend_cap_usd": spend_cap_usd,
        "cost_unknown_rows": state["cost_unknown_rows"],
        "error_counts_by_type": state["errors"],
        "per_slot_completed": state["per_slot"],
        "client_location": client_location,
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
        "git_source": git_source,
        "allow_dirty": allow_dirty,
        "provenance_tree_sha256": provenance_tree_sha256,
        "pool_path": str(Path(corpus_dir) / POOL_REL),
        "pool_size": len(pool),
        "pool_sha256": pool_sha256,
        "corpus_manifest_sha256": corpus_manifest_sha256,
        "schedule_sha256": schedule_sha256,
        "arrival_last_s": schedule[-1]["arrival_s"],
        "pool_passes": math.ceil(n_intents / len(pool)),
        "warmup_per_slot": warmup_n,
        "warmup_outcomes": warmup_outcomes,
        "trace_file": TRACE_FILE,
        "ledger_file": LEDGER_FILE,
        "order_file": ORDER_FILE if order_rows else None,
        "interpreter": {
            **manifest.get(interpreter_name, {}),
            "effective_base_url": getattr(slot_clients[0], "base_url", None),
        },
    }
    (out / INTEGRITY_FILE).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP-2026-003 RQ3 live load-trace runner")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run one interpreter/rate cell")
    run.add_argument("--corpus-dir", default=str(DEFAULT_CORPUS_DIR))
    run.add_argument("--interpreter", required=True, help="exact configs/ranbench/interpreters.json name")
    run.add_argument("--rate", required=True, type=float, help="Poisson arrival rate in intents/s")
    run.add_argument("--out", required=True)
    run.add_argument("--slots", type=int, default=DEFAULT_SLOTS)
    run.add_argument("--intents", type=int, default=DEFAULT_INTENTS, help="300 main; smaller values are smoke-only")
    run.add_argument("--seed", type=int, default=DEFAULT_SEED)
    run.add_argument("--spend-cap", type=float, default=20.0)
    run.add_argument("--client-location", help="required for hosted interpreters; recorded but not sent to providers")
    run.add_argument("--manifest")
    run.add_argument("--base-url", help="self-hosted server URL override")
    run.add_argument("--warmup", type=int, default=None, help="per-slot warm-ups; default 2 for AnyJev, else 0")
    run.add_argument("--allow-dirty", action="store_true")
    run.add_argument("--allow-stub", action="store_true")
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    if args.interpreter not in manifest:
        print(f"Unknown interpreter {args.interpreter!r}; available: {sorted(manifest)}", file=sys.stderr)
        return 2
    api_key = None
    if manifest[args.interpreter]["deployment"] == "hosted":
        api_key = load_openrouter_key()
        if not api_key:
            print("OPENROUTER_API_KEY not found (environment or key file).", file=sys.stderr)
            return 2
    try:
        report = run_rq3_load(
            corpus_dir=args.corpus_dir,
            interpreter_name=args.interpreter,
            rate=args.rate,
            out_dir=args.out,
            slots=args.slots,
            n_intents=args.intents,
            seed=args.seed,
            spend_cap_usd=args.spend_cap,
            client_location=args.client_location,
            manifest_path=args.manifest,
            api_key=api_key,
            base_url=args.base_url,
            allow_dirty=args.allow_dirty,
            allow_stub=args.allow_stub,
            warmup=args.warmup,
        )
    except (KeyError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    summary = {k: report[k] for k in (
        "run_id", "model", "rate_per_s", "expected_intents", "completed_intents", "valid_intents", "rho",
        "non_stationary", "total_spend_usd", "transport_error_rate", "rerun_block_required", "stop_reason",
        "complete",
    )}
    print(json.dumps(summary, indent=2))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
