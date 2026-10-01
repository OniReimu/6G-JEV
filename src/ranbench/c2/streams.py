"""Intent streams per design point (EXP-2026-003 protocol step 3).

One stream per design point: >= 100 events with Poisson issue times over a run of length
max(100 / rate, 10 x block length), drawn from the C2 pool, seeded, and written with its SHA-256 before any run.
Issue times are a Poisson process conditioned on its event count (sorted uniforms), so the run length and the
event count are both exactly as designed.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.edgebench.corpus.tuples import _stable_seed

BASE_SEED = 20260925
N_EVENTS_MIN = 100
MIN_BLOCKS = 10
T0_S = 2.0        # first possible issue time: traffic starts at 0.2 s, the xApp first runs at 1 s
TAIL_S = 10.0     # simulated after the run so the largest window W = 10 s of the last intent is complete


def run_length_s(rate: float, block_s: float) -> float:
    return max(N_EVENTS_MIN / rate, MIN_BLOCKS * block_s)


def n_events(rate: float, block_s: float) -> int:
    """100 events, plus the extra events at the same rate when 10 blocks outlast 100 / rate."""
    return max(N_EVENTS_MIN, round(rate * run_length_s(rate, block_s)))


def load_pool(path: str | Path) -> tuple[list[dict[str, Any]], str]:
    data = Path(path).read_bytes()
    pool = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    return pool, hashlib.sha256(data).hexdigest()


def load_completed_rq3_trace(path: str | Path, model: str, rate: float) -> list[dict[str, Any]]:
    """Load one complete frozen 300-decision RQ3 trace and return its first 100 arrivals."""
    path = Path(path)
    integrity_path = path.parent / "integrity.json"
    if not integrity_path.is_file():
        raise ValueError(f"{model}: RQ3 trace is missing adjacent integrity.json")
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    if (integrity.get("complete") is not True or integrity.get("model") != model
            or int(integrity.get("completed_intents", -1)) != 300
            or float(integrity.get("rate_per_s", -1.0)) != rate):
        raise ValueError(f"{model}: RQ3 trace integrity is not complete for this interpreter/rate")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if row.get("status") == "decision_recorded" and row.get("model") == model]
    rows.sort(key=lambda row: float(row["arrival_s"]))
    if len(rows) != 300:
        raise ValueError(f"{model}: complete RQ3 trace needs 300 decision_recorded rows, found {len(rows)}")
    selected = rows[:N_EVENTS_MIN]
    ids = [str(row["intent_id"]) for row in selected]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{model}: first 100 RQ3 trace intent_id values must be unique")
    return selected


def trace_sim_time_s(trace_rows: list[dict[str, Any]], block_s: float, n: int = N_EVENTS_MIN) -> float:
    """Exact replay stop time of the first n arrivals after six-decimal arrival-time normalisation."""
    if n < 1 or len(trace_rows) < n or block_s <= 0:
        raise ValueError(f"RQ3 trace needs at least {n} rows and a positive block length")
    rows = sorted(trace_rows, key=lambda row: float(row["arrival_s"]))[:n]
    a0 = float(rows[0]["arrival_s"])
    last_issue = round(T0_S + float(rows[-1]["arrival_s"]) - a0, 6)
    length = max(last_issue - T0_S + 1e-6, MIN_BLOCKS * block_s)
    return T0_S + length + TAIL_S


def _draw_order(n_pool: int, n: int, rng: np.random.Generator) -> list[int]:
    """Without replacement while the pool lasts; a fresh permutation for each further pass."""
    out: list[int] = []
    while len(out) < n:
        out.extend(int(i) for i in rng.permutation(n_pool))
    return out[:n]


def make_stream(
    pool: list[dict[str, Any]],
    rate: float,
    block_s: float,
    key: str,
    pool_sha256: str | None = None,
    base_seed: int = BASE_SEED,
) -> dict[str, Any]:
    """The stream of one design point. `key` names the design point (it seeds the draw)."""
    if rate <= 0 or block_s <= 0:
        raise ValueError("rate and block length must be positive")
    seed = _stable_seed(f"c2-stream:{key}", base_seed)
    rng = np.random.default_rng(seed)
    length = run_length_s(rate, block_s)
    n = n_events(rate, block_s)
    times = np.sort(T0_S + rng.uniform(0.0, length, size=n))
    events = []
    for j, (t, i) in enumerate(zip(times, _draw_order(len(pool), n, rng))):
        p = pool[i]
        events.append({
            "j": j,
            "intent_id": p["intent_id"],
            "t_issue_s": round(float(t), 6),
            "actuated": dict(p["actuated"]),
            "c2_class": p["c2_class"],
        })
    n_blocks = math.floor(length / block_s + 1e-9)
    return {
        "meta": {
            "key": key,
            "seed": seed,
            "base_seed": base_seed,
            "rate_per_s": rate,
            "block_s": block_s,
            "t0_s": T0_S,
            "run_length_s": length,
            "n_blocks": n_blocks,
            "sim_time_s": T0_S + length + TAIL_S,
            "n_events": n,
            "pool_size": len(pool),
            "pool_sha256": pool_sha256,
        },
        "events": events,
    }


def stream_from_trace(
    trace_rows: list[dict[str, Any]], pool: list[dict[str, Any]], block_s: float, key: str,
    pool_sha256: str | None = None, n: int = N_EVENTS_MIN,
) -> dict[str, Any]:
    """RQ3 replay: the load trace's first n intents with their arrival times (rows need intent_id, arrival_s)."""
    by_id = {p["intent_id"]: p for p in pool}
    rows = sorted(trace_rows, key=lambda r: r["arrival_s"])[:n]
    a0 = rows[0]["arrival_s"]
    events = []
    for j, r in enumerate(rows):
        p = by_id[r["intent_id"]]
        events.append({"j": j, "intent_id": p["intent_id"], "t_issue_s": round(T0_S + r["arrival_s"] - a0, 6),
                       "actuated": dict(p["actuated"]), "c2_class": p["c2_class"]})
    sim_time = trace_sim_time_s(rows, block_s, n=n)
    length = sim_time - T0_S - TAIL_S
    return {
        "meta": {"key": key, "seed": None, "base_seed": None, "rate_per_s": None, "block_s": block_s,
                 "t0_s": T0_S, "run_length_s": length, "n_blocks": math.floor(length / block_s + 1e-9),
                 "sim_time_s": T0_S + length + TAIL_S, "n_events": len(events), "pool_size": len(pool),
                 "pool_sha256": pool_sha256},
        "events": events,
    }


def write_stream(stream: dict[str, Any], path: str | Path) -> str:
    """Write stream JSON and <path>.sha256; refuse to overwrite a different existing stream (frozen before runs)."""
    path = Path(path)
    data = (json.dumps(stream, indent=1) + "\n").encode("utf-8")
    sha = hashlib.sha256(data).hexdigest()
    if path.exists() and path.read_bytes() != data:
        raise ValueError(f"{path} exists with different content; streams are frozen before any run")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    Path(str(path) + ".sha256").write_text(f"{sha}  {path.name}\n", encoding="utf-8")
    return sha


def load_stream(path: str | Path) -> dict[str, Any]:
    """Load a stream and verify it against its .sha256 file."""
    path = Path(path)
    data = path.read_bytes()
    want = Path(str(path) + ".sha256").read_text(encoding="utf-8").split()[0]
    got = hashlib.sha256(data).hexdigest()
    if got != want:
        raise ValueError(f"{path}: sha256 {got} != recorded {want}")
    return json.loads(data)
