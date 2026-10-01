"""Enforcement times and ns-3 schedules per arm (EXP-2026-003 design "Architecture modes", "Interpreter arms").

e_j = t_j + mode wait (periodic cycle, 4 interpretation slots) + decision latency + delta_A1 (modes P, E) + delta_E2.
Arms: L (true policy after the interpreter's latency), A (recorded policy after delta_E2), N (recorded policy after
its latency); controls oracle (true policy after delta_E2), fixed (true policy after a fixed latency d, through the
same mode path as L), no_update (empty schedule).
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.ranbench.corpus.spec import ACTUATABLE_ACTIONS, ACTUATED_FIELDS, CLUSTERS, load_config

DELTA_E2_S = 0.005        # spike B2 median of 2.5-6.9 ms (protocol step 3)
PUT_ACK_MEDIAN_S = 0.004300475120544434  # median a1_put_ack_latency_s over the 120 hosted C3 main records
HALF_POLL_S = 0.005       # half of the 10 ms A1 poll period
SLOTS = 4
LEVELS = ("critical", "high", "normal", "low")
NAMED_SCOPES = tuple(CLUSTERS) + ("all_cells",)
MODES = ("P-1", "P-10", "E", "X")
ARMS = ("L", "A", "N", "oracle", "fixed", "no_update")
SCHEDULE_HEADER = ("t_s", "scope", "class", "priority")


def delta_a1(put_ack_median_s: float, allow_override: bool = False) -> float:
    """delta_A1 = median A1 PUT-acknowledge time (C3) + half the poll period."""
    if not math.isclose(put_ack_median_s, PUT_ACK_MEDIAN_S, rel_tol=0.0, abs_tol=1e-12) and not allow_override:
        raise ValueError(
            f"A1 PUT-ack median must be pinned to {PUT_ACK_MEDIAN_S:.4f} s; "
            "use the explicit override flag to accept another value"
        )
    return put_ack_median_s + HALF_POLL_S


def class_map() -> dict[str, str]:
    return dict(load_config()["c2_class_map"])


def periodic_ready(t: np.ndarray, lat: np.ndarray, period: float, slots: int = SLOTS) -> np.ndarray:
    """Completion time of each interpretation under the periodic semantics (design.md P1-4).

    Intents issued during a cycle wait for the next cycle start (an intent issued exactly at a start joins that
    cycle); a batch runs through `slots` slots in issue order (each intent takes the earliest free slot); a batch
    running past the next grid start delays that cycle to the batch's completion (no overlap). NaN latency = no
    decision (occupies no slot, completion NaN).
    """
    n = len(t)
    done = np.full(n, np.nan)
    order = np.argsort(t, kind="stable")
    i = 0
    start = period * math.ceil(t[order[0]] / period - 1e-9) if n else 0.0
    while i < n:
        batch = []
        while i < n and t[order[i]] <= start + 1e-9:
            batch.append(order[i])
            i += 1
        if not batch:
            start = max(start, period * math.ceil(t[order[i]] / period - 1e-9))
            continue
        free = [start] * slots
        end = start
        for k in batch:
            if np.isnan(lat[k]):
                continue
            s = min(free)
            c = s + lat[k]
            free[free.index(s)] = c
            done[k] = c
            end = max(end, c)
        start = max(period * (math.floor(start / period + 1e-9) + 1), end)
    return done


def enforcement_times(
    mode: str, t: np.ndarray, lat: np.ndarray, delta_e2: float = DELTA_E2_S, d_a1: float | None = None
) -> np.ndarray:
    """e_j per mode; inf where there is no decision (NaN latency)."""
    t, lat = np.asarray(t, dtype=float), np.asarray(lat, dtype=float)
    if mode == "X":
        e = t + lat + delta_e2
    elif mode == "E" or mode.startswith("P-"):
        if d_a1 is None:
            raise ValueError(f"mode {mode} needs delta_A1")
        ready = t + lat if mode == "E" else periodic_ready(t, lat, float(mode[2:]))
        e = ready + d_a1 + delta_e2
    else:
        raise ValueError(f"unknown mode {mode}")
    return np.where(np.isnan(e), np.inf, e)


def install_of(policy: dict[str, Any] | None, cmap: dict[str, str]) -> tuple[str, str, str] | None:
    """What a policy installs in C2: (scope, c2 class, level|default), or None when it actuates nothing."""
    if not policy or policy.get("action") not in ACTUATABLE_ACTIONS:
        return None
    scope, cls = policy.get("scope"), cmap.get(policy.get("class"))
    if scope not in NAMED_SCOPES or cls is None:
        return None
    if policy["action"] == "revert_default":
        return scope, cls, "default"
    level = policy.get("priority")
    return (scope, cls, level) if level in LEVELS else None


def is_correct(policy: dict[str, Any] | None, truth: dict[str, str]) -> bool:
    return bool(policy) and all(policy.get(f) == truth[f] for f in ACTUATED_FIELDS)


def load_records(path: str | Path, model: str) -> dict[str, dict[str, Any]]:
    """Ledger-like rows of one interpreter keyed by case_id: labels, valid, latency_s (+ queue_wait_s for RQ3)."""
    out: dict[str, dict[str, Any]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("model") != model:
            continue
        if r["case_id"] in out:
            raise ValueError(f"{model}: two records for {r['case_id']} (one decision per intent)")
        out[r["case_id"]] = r
    return out


def _latency(rec: dict[str, Any]) -> float:
    lat = rec.get("latency_s")
    return float("nan") if lat is None else float(lat) + float(rec.get("queue_wait_s") or 0.0)


def build_arm(
    stream: dict[str, Any],
    arm: str,
    mode: str = "E",
    records: dict[str, dict[str, Any]] | None = None,
    fixed_d: float | None = None,
    delta_e2: float = DELTA_E2_S,
    d_a1: float | None = None,
    cmap: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Per-event enforcement records of one arm and its ns-3 schedule rows."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm}")
    cmap = cmap or class_map()
    ev = stream["events"]
    t = np.array([e["t_issue_s"] for e in ev], dtype=float)
    truths = [e["actuated"] for e in ev]
    if arm in ("L", "A", "N"):
        if records is None:
            raise ValueError(f"arm {arm} needs interpreter records")
        missing = sorted({e["intent_id"] for e in ev} - set(records))
        if missing:
            raise ValueError(f"no record for {len(missing)} intent(s), e.g. {missing[:3]}")
        recs = [records[e["intent_id"]] for e in ev]
        lat = np.array([_latency(r) for r in recs])
        recorded = [r.get("labels") if r.get("valid") else None for r in recs]

    if arm == "no_update":
        policies, e = [None] * len(ev), np.full(len(ev), np.inf)
        lat = np.full(len(ev), np.nan)
    elif arm == "oracle":
        policies, lat = truths, np.zeros(len(ev))
        e = t + delta_e2
    elif arm == "fixed":
        if fixed_d is None:
            raise ValueError("fixed arm needs fixed_d")
        policies, lat = truths, np.full(len(ev), float(fixed_d))
        e = enforcement_times(mode, t, lat, delta_e2, d_a1)
    elif arm == "L":
        policies, e = truths, enforcement_times(mode, t, lat, delta_e2, d_a1)
    elif arm == "A":
        policies, e = recorded, t + delta_e2
    else:  # N
        policies, e = recorded, enforcement_times(mode, t, lat, delta_e2, d_a1)

    sim_end = stream["meta"]["sim_time_s"]
    events, rows = [], []
    for k, (x, pol) in enumerate(zip(ev, policies)):
        inst = install_of(pol, cmap) if np.isfinite(e[k]) else None
        correct = arm != "no_update" and np.isfinite(e[k]) and is_correct(pol, x["actuated"])
        events.append({
            "j": x["j"], "intent_id": x["intent_id"], "t_issue_s": x["t_issue_s"],
            "latency_s": None if np.isnan(lat[k]) else float(lat[k]),
            "e_s": float(e[k]) if np.isfinite(e[k]) else None,
            "correct": bool(correct),
            "true_install": list(install_of(x["actuated"], cmap)),
            "install": list(inst) if inst else None,
            "in_schedule": bool(inst) and float(e[k]) <= sim_end,
        })
        if inst and float(e[k]) <= sim_end:
            rows.append((round(float(e[k]), 6), inst[0], inst[1], inst[2], x["j"]))
    rows.sort(key=lambda r: (r[0], r[4]))
    return {
        "meta": {"arm": arm, "mode": mode, "fixed_d": fixed_d, "delta_e2_s": delta_e2, "delta_a1_s": d_a1,
                 "stream_key": stream["meta"]["key"], "sim_time_s": sim_end},
        "events": events,
        "schedule": [r[:4] for r in rows],
    }


def write_schedule(rows: list[tuple], path: str | Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(SCHEDULE_HEADER)
        for t, scope, cls, level in rows:
            w.writerow((f"{t:.6f}", scope, cls, level))


def write_arm(arm: dict[str, Any], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_schedule(arm["schedule"], out / "schedule.csv")
    (out / "events.json").write_text(json.dumps({"meta": arm["meta"], "events": arm["events"]}, indent=1) + "\n",
                                     encoding="utf-8")
