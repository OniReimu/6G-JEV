"""Design-point-grouped makespan assignment for C2 tier-1/2 production rows.

A platform is one binary on one CPU type: ns-3's optimized profile builds with
-march=native and cross-platform byte identity is unproven (execution-log
2026-09-26 10:10 and 10:25), so C-4 pairing requires every row of one design
point to run on one platform. Groups are placed longest-first on the platform
whose resulting makespan is smallest (LPT over groups); within a platform, rows
run tier-first, then longest-first, on the earliest free slot.
"""
from __future__ import annotations

import csv
import fnmatch
import heapq
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from src.ranbench.c2.matrix import DEFAULT_PLATFORM_MULTIPLIERS

PLATFORMS = ("mac_m5_max", "mac_m4_max", "l40_cpu", "cluster_b")
DEFAULT_SLOTS = {"mac_m5_max": 12, "mac_m4_max": 12, "l40_cpu": 20, "cluster_b": 80}


@dataclass(frozen=True)
class Assignment:
    run_id: str
    priority_tier: int
    platform: str
    slot: int
    estimated_wall_hours: float
    estimated_start_hours: float
    estimated_finish_hours: float
    design_point: str = ""


def unassigned_reason(row: dict[str, str]) -> str:
    """Empty when the row is assignable now; otherwise why it waits."""
    if int(row["priority_tier"]) not in (1, 2) or row["rqs"] == "RQ5":
        return "rq5_or_tier3_not_assigned"
    if row["availability"] == "waiting_for_load_trace":
        return "waiting_for_load_trace"
    return ""


def _hours(row: dict[str, str], platform: str, multipliers: dict[str, float]) -> float:
    column = row.get(f"estimated_wall_hours_{platform}")
    if column:
        return float(column)
    if platform not in multipliers:
        raise ValueError(f"no wall-hour estimate or multiplier for platform {platform}")
    return float(row["estimated_wall_hours_mac_m4_max"]) * multipliers[platform]


def _schedule(rows: list[dict[str, str]], platform: str, n_slots: int,
              multipliers: dict[str, float]) -> list[Assignment]:
    order = sorted(rows, key=lambda row: (int(row["priority_tier"]),
                                          -_hours(row, platform, multipliers), row["run_id"]))
    free = [(0.0, slot) for slot in range(1, n_slots + 1)]
    out: list[Assignment] = []
    for row in order:
        start, slot = heapq.heappop(free)
        duration = _hours(row, platform, multipliers)
        heapq.heappush(free, (start + duration, slot))
        out.append(Assignment(
            run_id=row["run_id"], priority_tier=int(row["priority_tier"]), platform=platform,
            slot=slot, estimated_wall_hours=round(duration, 4),
            estimated_start_hours=round(start, 4), estimated_finish_hours=round(start + duration, 4),
            design_point=row["design_point"]))
    return out


def assign(rows: list[dict[str, str]], slots: dict[str, int],
           multipliers: dict[str, float] | None = None,
           pins: list[tuple[str, str]] | None = None) -> list[Assignment]:
    if not slots or any(value < 1 for value in slots.values()):
        raise ValueError("slots must give a positive count for every platform")
    factors = dict(DEFAULT_PLATFORM_MULTIPLIERS)
    factors.update(multipliers or {})
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if unassigned_reason(row):
            continue
        if not row.get("design_point"):
            raise ValueError(f"{row['run_id']}: matrix row has no design_point; regenerate the matrix")
        groups.setdefault(row["design_point"], []).append(row)

    # Fastest platform first on ties.
    speed_order = sorted(slots, key=lambda name: (factors.get(name, float("inf")), name))
    placed: dict[str, list[dict[str, str]]] = {name: [] for name in slots}
    plans: dict[str, list[Assignment]] = {name: [] for name in slots}
    pinned: dict[str, str] = {}
    for pattern, platform in pins or []:
        if platform not in slots:
            raise ValueError(f"unknown pin platform {platform}")
        matches = [key for key in groups if fnmatch.fnmatchcase(key, pattern)]
        if not matches:
            raise ValueError(f"pin glob {pattern!r} matched no design point")
        for key in matches:
            if key in pinned and pinned[key] != platform:
                raise ValueError(
                    f"design point {key} pinned to both {pinned[key]} and {platform}")
            pinned[key] = platform
    for key, platform in sorted(pinned.items()):
        placed[platform].extend(groups[key])
    for platform in speed_order:
        if placed[platform]:
            plans[platform] = _schedule(placed[platform], platform, slots[platform], factors)

    fastest = speed_order[0]
    work = {
        key: (max(_hours(row, fastest, factors) for row in members),
              sum(_hours(row, fastest, factors) for row in members))
        for key, members in groups.items()
    }
    order = sorted((key for key in groups if key not in pinned),
                   key=lambda name: (-work[name][0], -work[name][1], name))
    for key in order:
        best: tuple[float, int, str, list[Assignment]] | None = None
        members = groups[key]
        for rank, platform in enumerate(speed_order):
            trial = _schedule(placed[platform] + groups[key], platform, slots[platform], factors)
            makespan = max(item.estimated_finish_hours for item in trial)
            if best is None or (makespan, rank) < best[:2]:
                best = (makespan, rank, platform, trial)
        assert best is not None
        placed[best[2]].extend(groups[key])
        plans[best[2]] = best[3]
    assignments = [item for name in speed_order for item in plans[name]]
    for key in groups:
        where = {item.platform for item in assignments if item.design_point == key}
        if len(where) != 1:
            raise AssertionError(f"design point {key} split across platforms {sorted(where)}")
    return assignments


def write_assignment(assignments: list[Assignment], out_dir: str | Path,
                     rows: list[dict[str, str]] | None = None) -> dict[str, object]:
    out = Path(out_dir)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite assignment directory: {out}")
    out.mkdir(parents=True)
    fields = list(Assignment.__dataclass_fields__)
    with (out / "assignment.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: getattr(item, field) for field in fields} for item in assignments])
    summary: dict[str, object] = {"rows": len(assignments), "platforms": {}}
    platforms = list(dict.fromkeys([*PLATFORMS, *(item.platform for item in assignments)]))
    for platform in platforms:
        items = sorted((item for item in assignments if item.platform == platform),
                       key=lambda item: (item.estimated_start_hours, item.slot))
        # Queue order = planned start order (tier first, then longest first).
        (out / f"run-list-{platform}.txt").write_text(
            "".join(f"{item.run_id}\n" for item in items), encoding="utf-8")
        summary["platforms"][platform] = {
            "rows": len(items),
            "design_points": sorted({item.design_point for item in items}),
            "core_hours": round(sum(item.estimated_wall_hours for item in items), 3),
            "makespan_hours": round(max((item.estimated_finish_hours for item in items), default=0.0), 3),
        }
    summary["overall_makespan_hours"] = max(
        value["makespan_hours"] for value in summary["platforms"].values())
    if rows is not None:
        summary["unassigned"] = dict(sorted(Counter(
            reason for reason in map(unassigned_reason, rows) if reason).items()))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
