"""Frozen C2 production matrix and full-fidelity cost model for EXP-2026-003.

The matrix contains one row per ns-3 process. Paths name immutable inputs that
are materialised after the pilot fixes the block length and after interpreter
load traces exist.
"""
from __future__ import annotations

import csv
import json
import math
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from src.ranbench.c2.streams import (
    T0_S,
    TAIL_S,
    load_completed_rq3_trace,
    run_length_s,
    trace_sim_time_s,
)

INTERPRETERS = (
    "Jev-1.13.0",
    "DeepSeek-V4.1-Flash",
    "GLM-5.3-Flash",
    "Qwen3.8-Flash",
    "SemIf-Qwen3.5-4B",
    "AnyJev-L0",
    "Qwen3.5-4B-JSON",
)
CONTROL_ARMS = ("oracle", "fixed-0.1", "fixed-1", "fixed-5", "no-update")
RQ2_RATES = (0.1, 0.3, 1.0)
RQ2_SPEEDS = (3.0, 60.0, 120.0)
RQ3_RATES = (0.1, 0.5, 1.0, 2.0)

BASE_UES = 105
BASE_MAC_S_PER_UE_S = 0.89
DEFAULT_PLATFORM_MULTIPLIERS = {
    "mac_m4_max": 1.0,
    "mac_m5_max": 0.85,
    # 369.6606 wall-s / (105 UEs * 2 sim-s) / 0.89 M4 wall-s/UE-s.
    "l40_cpu": 1.977852327447833,
    "cluster_b": 2.2,
}


def slug(value: str | float) -> str:
    """Stable path-safe token used by run IDs and input filenames."""
    text = f"{value:g}" if isinstance(value, float) else str(value)
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def rate_tag(rate: float) -> str:
    return f"{rate:g}".replace(".", "p")


def simulated_seconds(rate: float, block_s: float) -> float:
    """ns-3 stop time, including the 2 s lead-in and 10 s outcome tail."""
    return T0_S + run_length_s(rate, block_s) + TAIL_S


def wall_hours(sim_s: float, ues: int, multiplier: float = 1.0,
               mac_s_per_ue_s: float = BASE_MAC_S_PER_UE_S) -> float:
    """Single-core wall-hours under the measured cost and total cost ~ UE^1.5."""
    scale = math.sqrt(ues / BASE_UES)
    return mac_s_per_ue_s * ues * sim_s * scale * multiplier / 3600.0


@dataclass(frozen=True)
class MatrixRow:
    run_id: str
    rqs: str
    priority_tier: int
    availability: str
    mode: str
    arm: str
    interpreter: str
    rate_per_s: float
    speed_kmh: float
    ues_per_cell: int
    ues: int
    simulated_seconds: float
    stream_file: str
    schedule_file: str
    config_file: str
    rng_run: int
    estimated_wall_hours_mac_m4_max: float
    estimated_wall_hours_mac_m5_max: float
    estimated_wall_hours_l40_cpu: float
    estimated_wall_hours_cluster_b: float
    notes: str = ""
    reused_arm_a_run_id: str = ""
    reused_arm_c_run_id: str = ""
    # Rows that share one stream, config and UE count form one design point; C-4 pairing
    # requires all of them on one platform (execution-log 2026-09-26 10:10).
    design_point: str = ""


def _row(*, input_root: Path, rqs: str, tier: int, mode: str, arm: str,
         interpreter: str, rate: float, speed: float, ues_per_cell: int,
         block_s: float, run_id: str, availability: str = "ready_after_pilot",
         notes: str = "", platform_multipliers: dict[str, float],
         sim_time_override: float | None = None, reused_arm_a_run_id: str = "",
         reused_arm_c_run_id: str = "") -> MatrixRow:
    ues = 21 * ues_per_cell
    sim_s = simulated_seconds(rate, block_s) if sim_time_override is None else sim_time_override
    stream_key = f"rate_{rate_tag(rate)}_speed_{slug(speed)}"
    design_point = f"dp_r{rate_tag(rate)}_s{slug(speed)}_u{ues_per_cell}"
    if rqs == "RQ3":
        stream_key = f"rq3_{slug(interpreter)}_rate_{rate_tag(rate)}"
        design_point = f"dp_rq3_{slug(interpreter)}_r{rate_tag(rate)}_u{ues_per_cell}"
    elif rqs == "RQ5":
        stream_key = "rq5_base"
    estimates = {
        name: round(wall_hours(sim_s, ues, factor), 4)
        for name, factor in platform_multipliers.items()
    }
    return MatrixRow(
        run_id=run_id,
        rqs=rqs,
        priority_tier=tier,
        availability=availability,
        mode=mode,
        arm=arm,
        interpreter=interpreter,
        rate_per_s=rate,
        speed_kmh=speed,
        ues_per_cell=ues_per_cell,
        ues=ues,
        simulated_seconds=round(sim_s, 6),
        stream_file=str(input_root / "streams" / f"{stream_key}.json"),
        schedule_file=str(input_root / "schedules" / f"{run_id}.csv"),
        config_file=str(input_root / "configs" / f"speed_{slug(speed)}_ues_{ues_per_cell}.json"),
        rng_run=1,
        estimated_wall_hours_mac_m4_max=estimates["mac_m4_max"],
        estimated_wall_hours_mac_m5_max=estimates["mac_m5_max"],
        estimated_wall_hours_l40_cpu=estimates["l40_cpu"],
        estimated_wall_hours_cluster_b=estimates["cluster_b"],
        notes=notes,
        reused_arm_a_run_id=reused_arm_a_run_id,
        reused_arm_c_run_id=reused_arm_c_run_id,
        design_point=design_point,
    )


def generate_matrix(input_root: str | Path, block_s: float = 10.0,
                    platform_multipliers: dict[str, float] | None = None,
                    rq3_trace_paths: dict[str, str | Path] | None = None) -> list[MatrixRow]:
    """Return every frozen C2 simulation in priority order."""
    if block_s <= 0:
        raise ValueError("block_s must be positive")
    factors = dict(DEFAULT_PLATFORM_MULTIPLIERS)
    if platform_multipliers:
        factors.update(platform_multipliers)
    if set(factors) != set(DEFAULT_PLATFORM_MULTIPLIERS) or any(v <= 0 for v in factors.values()):
        raise ValueError(f"platform multipliers must be positive values for {sorted(DEFAULT_PLATFORM_MULTIPLIERS)}")
    root = Path(input_root)
    traces = dict(rq3_trace_paths or {})
    rows: list[MatrixRow] = []

    # Tier 1: RQ2 L arms and controls at all 3 x 3 points.
    for rate in RQ2_RATES:
        for speed in RQ2_SPEEDS:
            for interpreter in INTERPRETERS:
                rid = f"rq2_e_r{rate_tag(rate)}_s{slug(speed)}_l_{slug(interpreter)}"
                rows.append(_row(input_root=root, rqs="RQ2", tier=1, mode="E", arm="L",
                                 interpreter=interpreter, rate=rate, speed=speed, ues_per_cell=5,
                                 block_s=block_s, run_id=rid,
                                 availability="waiting_for_c1_ledger",
                                 platform_multipliers=factors))
            for arm in CONTROL_ARMS:
                rid = f"rq2_e_r{rate_tag(rate)}_s{slug(speed)}_{slug(arm)}"
                rows.append(_row(input_root=root, rqs="RQ2", tier=1, mode="E", arm=arm,
                                 interpreter="control", rate=rate, speed=speed, ues_per_cell=5,
                                 block_s=block_s, run_id=rid, platform_multipliers=factors))

    # D-4 retained the A/N tier at every RQ2 point; it runs after the tier-1 rows.
    for rate in RQ2_RATES:
        for speed in RQ2_SPEEDS:
            for arm in ("A", "N"):
                for interpreter in INTERPRETERS:
                    rid = (f"rq2_e_r{rate_tag(rate)}_s{slug(speed)}_"
                           f"{arm.lower()}_{slug(interpreter)}")
                    rows.append(_row(
                        input_root=root, rqs="RQ2", tier=2, mode="E", arm=arm,
                        interpreter=interpreter, rate=rate, speed=speed, ues_per_cell=5,
                        block_s=block_s, run_id=rid,
                        availability="waiting_for_c1_ledger",
                        platform_multipliers=factors,
                        notes="exploratory A/N decomposition",
                    ))

    # Base-point controls (C-1/C-2 at every design point; H1 is stated at the base point;
    # RQ5 arm (c) reuses the oracle row). The base point is not on the RQ2 grid.
    # Controls are interpreter- and mode-independent, so these rows also serve P-1/P-10.
    for arm in CONTROL_ARMS:
        rid = f"rq1_e_r0p3_s30_{slug(arm)}"
        rows.append(_row(input_root=root, rqs="RQ1", tier=1, mode="E", arm=arm,
                         interpreter="control", rate=0.3, speed=30.0, ues_per_cell=5,
                         block_s=block_s, run_id=rid, platform_multipliers=factors))

    # RQ1 E is tier 1. Periodic modes are tier 2.
    for mode, tier in (("E", 1), ("P-1", 2), ("P-10", 2)):
        for interpreter in INTERPRETERS:
            rid = f"rq1_{slug(mode)}_r0p3_s30_l_{slug(interpreter)}"
            rows.append(_row(input_root=root, rqs="RQ1", tier=tier, mode=mode, arm="L",
                             interpreter=interpreter, rate=0.3, speed=30.0, ues_per_cell=5,
                             block_s=block_s, run_id=rid,
                             availability="waiting_for_c1_ledger",
                             platform_multipliers=factors))

    # Registered lower-priority A/N decomposition at the base point.
    for arm in ("A", "N"):
        for interpreter in INTERPRETERS:
            rid = f"rq2_e_r0p3_s30_{arm.lower()}_{slug(interpreter)}"
            rows.append(_row(input_root=root, rqs="RQ2", tier=2, mode="E", arm=arm,
                             interpreter=interpreter, rate=0.3, speed=30.0, ues_per_cell=5,
                             block_s=block_s, run_id=rid,
                             availability="waiting_for_c1_ledger",
                             platform_multipliers=factors,
                             notes="exploratory A/N decomposition at base point"))

    # RQ3 replays use each interpreter's net load trace; the scale point is additional.
    for interpreter in INTERPRETERS:
        for rate in RQ3_RATES:
            trace_key = f"{interpreter}@{rate:g}"
            trace_time = None
            availability = "waiting_for_load_trace"
            if trace_key in traces:
                trace_rows = load_completed_rq3_trace(traces[trace_key], interpreter, rate)
                trace_time = trace_sim_time_s(trace_rows, block_s)
                availability = "ready_after_pilot"
            rid = f"rq3_e_r{rate_tag(rate)}_s30_n_{slug(interpreter)}"
            rows.append(_row(input_root=root, rqs="RQ3", tier=2, mode="E", arm="N",
                             interpreter=interpreter, rate=rate, speed=30.0, ues_per_cell=5,
                             block_s=block_s, run_id=rid, availability=availability,
                             notes="first 100 arrivals from the four-slot live load trace",
                             platform_multipliers=factors, sim_time_override=trace_time))
        trace_key = f"{interpreter}@1"
        trace_time = None
        availability = "waiting_for_load_trace"
        if trace_key in traces:
            trace_rows = load_completed_rq3_trace(traces[trace_key], interpreter, 1.0)
            trace_time = trace_sim_time_s(trace_rows, block_s)
            availability = "ready_after_pilot"
        rid = f"rq3_scale_e_r1_s30_n_{slug(interpreter)}"
        rows.append(_row(input_root=root, rqs="RQ3", tier=2, mode="E", arm="N",
                         interpreter=interpreter, rate=1.0, speed=30.0, ues_per_cell=20,
                         block_s=block_s, run_id=rid, availability=availability,
                         notes="21 cells x 20 UEs/cell scale point", platform_multipliers=factors,
                         sim_time_override=trace_time))

    # RQ5: only arm (b) is a new process. Each row points to the existing base-point
    # N arm (a) and the single existing base-point oracle control (c), so neither is duplicated.
    oracle_reference = "rq1_e_r0p3_s30_oracle"
    for interpreter in INTERPRETERS:
        rid = f"rq5_e_r0p3_s30_b_requery_1s_{slug(interpreter)}"
        n_reference = f"rq2_e_r0p3_s30_n_{slug(interpreter)}"
        rows.append(_row(
            input_root=root, rqs="RQ5", tier=3, mode="E", arm="b-requery-1s",
            interpreter=interpreter, rate=0.3, speed=30.0, ues_per_cell=5,
            block_s=block_s, run_id=rid, availability="waiting_for_c1_ledger_and_pilot",
            notes="new arm (b); arms (a)/(c) are zero-incremental-cost references",
            platform_multipliers=factors, reused_arm_a_run_id=n_reference,
            reused_arm_c_run_id=oracle_reference,
        ))

    seen: set[str] = set()
    for row in rows:
        if row.run_id in seen:
            raise AssertionError(f"duplicate run_id {row.run_id}")
        seen.add(row.run_id)
    for row in rows:
        for reference in (row.reused_arm_a_run_id, row.reused_arm_c_run_id):
            if reference and reference not in seen:
                raise AssertionError(f"{row.run_id} reuses missing run {reference}")
    return sorted(rows, key=lambda row: (row.priority_tier, row.run_id))


def write_matrix(rows: Iterable[MatrixRow], path: str | Path) -> None:
    """Write CSV or JSON by suffix, refusing to replace an existing file."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite matrix: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [asdict(row) for row in rows]
    if path.suffix == ".json":
        path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        return
    if path.suffix != ".csv":
        raise ValueError("matrix output must end in .csv or .json")
    if not records:
        raise ValueError("cannot write an empty matrix")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def totals(rows: Iterable[MatrixRow]) -> dict[str, object]:
    records = list(rows)
    platforms = {
        name: round(sum(getattr(row, f"estimated_wall_hours_{name}") for row in records), 3)
        for name in DEFAULT_PLATFORM_MULTIPLIERS
    }
    return {
        "rows": len(records),
        "rows_by_tier": {str(tier): sum(row.priority_tier == tier for row in records) for tier in (1, 2, 3)},
        "ue_seconds": round(sum(row.ues * row.simulated_seconds for row in records), 3),
        "estimated_core_hours_if_all_on_platform": platforms,
    }
