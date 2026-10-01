#!/usr/bin/env python3
"""Freeze the completed C3 path and RQ3 interpreter-load analyses.

The script reads the raw C3 records and RQ3 load traces (see the experiment READMEs for where they go),
reuses the experiment implementation's queue/utilisation/enforcement definitions,
and writes deterministic CSVs into the experiment results.  It also checks the rounded
values recorded when the runs completed and exits non-zero on any mismatch. Run it from the repository root.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
C3_OUT = Path("experiments/c3-real-stack/results")
RQ3_OUT = Path("experiments/c1-interpretation/results/rq3-load")
JFC = ROOT
C3_ROOT = Path("experiments/c3-real-stack/results/runs")
RQ3_ROOT = Path("experiments/c1-interpretation/results/rq3-load/traces")

# The listed roots are the complete frozen RQ3 input set, in protocol/session order.
RQ3_SESSIONS = (
    "main-hosted-v1",
    "main-hosted-v1-qwen-offpeak",
    "main-hosted-v1-qwen-offpeak2",
    "selfhosted/main-v1",
    "selfhosted/main-v1b",
    "selfhosted/main-v1c",
    "selfhosted-d12-idle",
)
# D-12 addendum: the idle-node replication replaces the main-v1b AnyJev-L0 cells as the primary cells.
PRIMARY_SESSION = {"AnyJev-L0": "selfhosted-d12-idle"}

MODELS = (
    "Jev-1.13.0",
    "SemIf-Qwen3.5-4B",
    "AnyJev-L0",
    "DeepSeek-V4.1-Flash",
    "GLM-5.3-Flash",
    "Qwen3.8-Flash",
    "Qwen3.5-4B-JSON",
)
HOSTED = {
    "Jev-1.13.0",
    "DeepSeek-V4.1-Flash",
    "GLM-5.3-Flash",
    "Qwen3.8-Flash",
}
RATES = (0.1, 0.5, 1.0, 2.0)

# Import the experiment's definitions instead of duplicating them here.
sys.path.insert(0, str(JFC))
from src.ranbench.c2.queueing import utilisation  # noqa: E402
from src.ranbench.c2.schedule import (  # noqa: E402
    DELTA_E2_S,
    PUT_ACK_MEDIAN_S,
    _latency,
    delta_a1,
    enforcement_times,
    install_of,
)
from src.ranbench.corpus.spec import load_config  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSON-lines file, rejecting non-object rows."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def numeric(rows: Iterable[dict[str, Any]], field: str) -> np.ndarray:
    values = [float(row[field]) for row in rows if isinstance(row.get(field), (int, float))]
    return np.asarray(values, dtype=float)


def quantile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values, q, method="linear"))


def rtt_fields(model: str) -> dict[str, Any]:
    tags = {
        "SemIf-Qwen3.5-4B": "semif",
        "AnyJev-L0": "anyjev",
        "Qwen3.5-4B-JSON": "qwen_json",
    }
    if model not in tags:
        return {
            "tunnel_rtt_before_median_ms": float("nan"),
            "tunnel_rtt_after_median_ms": float("nan"),
            "tunnel_rtt_note": "not applicable (hosted interpreter)",
            "source_rtt": "",
        }
    path = C3_ROOT / "selfhosted-rtt" / f"{tags[model]}.jsonl"
    rows = read_jsonl(path)
    by_phase = {str(row["phase"]): row for row in rows}
    if set(by_phase) != {"before", "after"}:
        raise ValueError(f"{path}: expected exactly before/after RTT phases")
    note = "measured around main run"
    if model == "SemIf-Qwen3.5-4B":
        note = "same-node/path pre-rerun measurement; execution-log flag"
    return {
        "tunnel_rtt_before_median_ms": float(by_phase["before"]["median_ms"]),
        "tunnel_rtt_after_median_ms": float(by_phase["after"]["median_ms"]),
        "tunnel_rtt_note": note,
        "source_rtt": str(path),
    }


def analyse_c3() -> pd.DataFrame:
    run_dirs = sorted(path.parent for path in C3_ROOT.glob("c3main-*/records.jsonl"))
    if len(run_dirs) != len(MODELS):
        raise ValueError(f"expected {len(MODELS)} C3 main runs, found {len(run_dirs)}")
    rows_out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for run_dir in run_dirs:
        records_path = run_dir / "records.jsonl"
        integrity_path = run_dir / "integrity.json"
        records = read_jsonl(records_path)
        integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
        if integrity.get("status") != "PASS":
            raise ValueError(f"{integrity_path}: integrity is not PASS")
        models = {str(row.get("interpreter")) for row in records}
        if len(models) != 1:
            raise ValueError(f"{records_path}: expected one interpreter, found {sorted(models)}")
        model = next(iter(models))
        if model not in MODELS or model in seen:
            raise ValueError(f"unexpected or duplicate C3 interpreter: {model}")
        seen.add(model)
        if len(records) != 30 or integrity.get("actual_decisions") != 30:
            raise ValueError(f"{records_path}: expected 30 decisions")

        ell = numeric(records, "ell_s")
        da1 = numeric(records, "delta_a1_s")
        de2 = numeric(records, "delta_e2_s")
        kpm = numeric(records, "kpm_change_upper_bound_s")
        row = {
            "model": model,
            "deployment": "hosted" if model in HOSTED else "self-hosted",
            "ell_median_s": quantile(ell, 0.50),
            "ell_p95_s": quantile(ell, 0.95),
            "delta_a1_median_ms": 1000.0 * quantile(da1, 0.50),
            "delta_a1_p95_ms": 1000.0 * quantile(da1, 0.95),
            "delta_e2_median_ms": 1000.0 * quantile(de2, 0.50),
            "delta_e2_p95_ms": 1000.0 * quantile(de2, 0.95),
            "kpm_upper_bound_median_s": quantile(kpm, 0.50),
            "kpm_upper_bound_p95_s": quantile(kpm, 0.95),
            "n": len(records),
            "n_delta_a1": int(da1.size),
            "n_delta_e2": int(de2.size),
            "n_kpm_observed": int(kpm.size),
            "kpm_null_count": len(records) - int(kpm.size),
            "source_records": str(records_path),
            "source_integrity": str(integrity_path),
        }
        row.update(rtt_fields(model))
        rows_out.append(row)
    if seen != set(MODELS):
        raise ValueError(f"C3 roster mismatch: missing={sorted(set(MODELS) - seen)}")
    frame = pd.DataFrame(rows_out)
    frame["model"] = pd.Categorical(frame["model"], MODELS, ordered=True)
    return frame.sort_values("model").reset_index(drop=True).assign(model=lambda x: x["model"].astype(str))


def block_number(path: Path) -> int:
    try:
        return int(path.parent.name.rsplit("__block", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"cannot parse block number from {path}") from exc


def analyse_rq3_availability() -> tuple[pd.DataFrame, dict[tuple[str, float], Path]]:
    records: list[dict[str, Any]] = []
    for session_index, session in enumerate(RQ3_SESSIONS):
        session_root = RQ3_ROOT / session
        if not session_root.is_dir():
            raise FileNotFoundError(session_root)
        for integrity_path in sorted(session_root.rglob("integrity.json")):
            run_dir = integrity_path.parent
            trace_path, ledger_path = run_dir / "trace.jsonl", run_dir / "ledger.jsonl"
            if not trace_path.is_file() or not ledger_path.is_file():
                raise FileNotFoundError(f"{run_dir}: missing trace.jsonl or ledger.jsonl")
            integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
            trace = read_jsonl(trace_path)
            model, rate = str(integrity["model"]), float(integrity["rate_per_s"])
            if model not in MODELS or rate not in RATES:
                raise ValueError(f"{integrity_path}: unexpected model/rate {model}/{rate}")
            completed = int(integrity["completed_intents"])
            if len(trace) != completed:
                raise ValueError(f"{trace_path}: {len(trace)} rows != completed_intents {completed}")
            http_429_count = sum(row.get("http_status") == 429 for row in trace)
            records.append(
                {
                    "model": model,
                    "deployment": str(integrity["deployment"]),
                    "rate_per_s": rate,
                    "session": session,
                    "session_order": session_index,
                    "block": block_number(integrity_path),
                    "session_start_utc": str(integrity["started_at_utc"]),
                    "http_429_count": http_429_count,
                    "http_429_share": http_429_count / completed if completed else float("nan"),
                    "n": completed,
                    "complete": bool(integrity["complete"]),
                    "primary": False,
                    "source_trace": str(trace_path),
                    "source_ledger": str(ledger_path),
                    "source_integrity": str(integrity_path),
                }
            )
    availability = pd.DataFrame(records).sort_values(
        ["session_order", "model", "rate_per_s", "block"], kind="stable"
    ).reset_index(drop=True)
    primary: dict[tuple[str, float], Path] = {}
    for key, sub in availability.groupby(["model", "rate_per_s"], sort=False):
        complete = sub[sub["complete"]]
        if key[0] in PRIMARY_SESSION:
            complete = complete[complete["session"] == PRIMARY_SESSION[key[0]]]
        if complete.empty:
            raise ValueError(f"no complete RQ3 block for {key}")
        idx = int(complete.index[0])
        availability.loc[idx, "primary"] = True
        primary[(str(key[0]), float(key[1]))] = Path(str(availability.loc[idx, "source_integrity"])).parent
    expected = {(model, rate) for model in MODELS for rate in RATES}
    if set(primary) != expected:
        raise ValueError(f"RQ3 primary-cell mismatch: missing={sorted(expected - set(primary))}")
    availability = availability.drop(columns=["session_order"])
    return availability, primary


def analyse_rq3_load(primary: dict[tuple[str, float], Path]) -> pd.DataFrame:
    cfg = load_config(JFC / "data" / "ranbench" / "ranintent-v1" / "config.json")
    cmap = dict(cfg["c2_class_map"])
    d_a1 = delta_a1(PUT_ACK_MEDIAN_S)
    rows_out: list[dict[str, Any]] = []
    for model in MODELS:
        for rate in RATES:
            run_dir = primary[(model, rate)]
            integrity_path, trace_path = run_dir / "integrity.json", run_dir / "trace.jsonl"
            integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
            trace = read_jsonl(trace_path)
            if len(trace) != 300 or integrity.get("complete") is not True:
                raise ValueError(f"{run_dir}: primary RQ3 block is not a complete 300-row trace")
            services = numeric(trace, "service_latency_s")
            waits = numeric(trace, "queue_wait_s")
            if services.size != len(trace) or waits.size != len(trace):
                raise ValueError(f"{trace_path}: primary trace has missing service/queue values")
            util = utilisation(rate, services, int(integrity["slots"]))
            if not math.isclose(float(util["rho"]), float(integrity["rho"]), rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"{integrity_path}: rho does not reproduce from service times")
            if bool(util["non_stationary"]) != bool(integrity["non_stationary"]):
                raise ValueError(f"{integrity_path}: non_stationary does not reproduce")

            total_latency = np.asarray([_latency(row) for row in trace], dtype=float)
            enforced_at = enforcement_times(
                "E", np.zeros(len(trace), dtype=float), total_latency,
                delta_e2=DELTA_E2_S, d_a1=d_a1,
            )
            actuatable = np.asarray(
                [install_of(row.get("policy") if row.get("valid") else None, cmap) is not None for row in trace],
                dtype=bool,
            )
            enforced_within_1s = actuatable & np.isfinite(enforced_at) & (enforced_at <= 1.0)
            rows_out.append(
                {
                    "model": model,
                    "deployment": "hosted" if model in HOSTED else "self-hosted",
                    "rate_per_s": rate,
                    "rho": float(util["rho"]),
                    "queue_wait_p50_s": quantile(waits, 0.50),
                    "queue_wait_p95_s": quantile(waits, 0.95),
                    "queue_wait_p99_s": quantile(waits, 0.99),
                    "share_enforced_within_1s": float(np.mean(enforced_within_1s)),
                    "non_stationary": bool(util["non_stationary"]),
                    "n": len(trace),
                    "delta_a1_s": d_a1,
                    "delta_e2_s": DELTA_E2_S,
                    "source_trace": str(trace_path),
                    "source_integrity": str(integrity_path),
                }
            )
    return pd.DataFrame(rows_out)


def rounded_sequence(frame: pd.DataFrame, model: str) -> str:
    sub = frame[frame["model"] == model].sort_values("rate_per_s")
    return "/".join(f"{value:.3f}" for value in sub["rho"])


def compare_execution_log(c3: pd.DataFrame, rq3: pd.DataFrame) -> list[str]:
    """Compare against the numeric checkpoints recorded on 2026-09-26/27."""
    failures: list[str] = []
    lines: list[str] = []
    expected_ell = {
        "Jev-1.13.0": 0.286,
        "SemIf-Qwen3.5-4B": 0.357,
        "Qwen3.5-4B-JSON": 0.432,
        "DeepSeek-V4.1-Flash": 0.527,
        "GLM-5.3-Flash": 1.055,
        "AnyJev-L0": 1.068,
        "Qwen3.8-Flash": 2.352,
    }
    for model, expected in expected_ell.items():
        actual = float(c3.loc[c3["model"] == model, "ell_median_s"].iloc[0])
        ok = round(actual, 3) == expected
        lines.append(f"C3 ell median {model}: {actual:.3f} s vs log {expected:.3f} s [{'OK' if ok else 'FAIL'}]")
        if not ok:
            failures.append(lines[-1])
    for column, label, decimals, expected in (
        ("delta_a1_median_ms", "delta_A1 median range", 1, (15.5, 19.1)),
        ("delta_e2_median_ms", "delta_E2 median range", 1, (2.4, 3.1)),
        ("kpm_upper_bound_median_s", "KPM upper-bound median range", 2, (1.25, 1.45)),
    ):
        actual = (round(float(c3[column].min()), decimals), round(float(c3[column].max()), decimals))
        ok = actual == expected
        lines.append(f"C3 {label}: {actual[0]:.{decimals}f}-{actual[1]:.{decimals}f} vs log "
                     f"{expected[0]:.{decimals}f}-{expected[1]:.{decimals}f} [{'OK' if ok else 'FAIL'}]")
        if not ok:
            failures.append(lines[-1])

    expected_rho = {
        "SemIf-Qwen3.5-4B": "0.011/0.040/0.095/0.261",
        "AnyJev-L0": "0.017/0.119/0.347/1.169",  # D-12 idle replication (log 2026-09-29)
        "Qwen3.5-4B-JSON": "0.010/0.052/0.106/0.217",
        "Qwen3.8-Flash": "0.059/0.277/0.524/1.085",
    }
    for model, expected in expected_rho.items():
        actual = rounded_sequence(rq3, model)
        ok = actual == expected
        lines.append(f"RQ3 rho {model} at 0.1/0.5/1/2 s^-1: {actual} vs log {expected} "
                     f"[{'OK' if ok else 'FAIL'}]")
        if not ok:
            failures.append(lines[-1])
    print("\nExecution-log comparison")
    print("------------------------")
    print("\n".join(lines))
    return failures


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.12g")


def main() -> int:
    c3 = analyse_c3()
    availability, primary = analyse_rq3_availability()
    rq3 = analyse_rq3_load(primary)
    failures = compare_execution_log(c3, rq3)
    if failures:
        print("\nSTOP: one or more frozen execution-log values did not reproduce.", file=sys.stderr)
        return 1
    write_csv(c3, C3_OUT / "c3_path_decomposition.csv")
    write_csv(rq3, RQ3_OUT / "rq3_load.csv")
    write_csv(availability, RQ3_OUT / "rq3_availability.csv")
    print(f"\nWrote {len(c3)} C3 rows, {len(rq3)} RQ3 primary rows, and "
          f"{len(availability)} RQ3 availability rows to {C3_OUT} and {RQ3_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
