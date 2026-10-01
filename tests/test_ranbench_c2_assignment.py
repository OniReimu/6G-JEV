from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.ranbench.c2.assignment import DEFAULT_SLOTS, PLATFORMS, assign, write_assignment
from src.ranbench.c2.matrix import generate_matrix, write_matrix


def _records(tmp_path: Path) -> list[dict[str, str]]:
    matrix = tmp_path / "matrix.csv"
    write_matrix(generate_matrix(tmp_path / "inputs"), matrix)
    with matrix.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_assignment_keeps_each_design_point_on_one_platform(tmp_path: Path) -> None:
    records = _records(tmp_path)
    assignments = assign(records, dict(DEFAULT_SLOTS))
    expected = {row["run_id"] for row in records
                if row["priority_tier"] in ("1", "2") and row["rqs"] != "RQ5"
                and row["availability"] != "waiting_for_load_trace"}
    assert {item.run_id for item in assignments} == expected
    assert len(assignments) == len(expected) == 274
    platform_of: dict[str, set[str]] = {}
    for item in assignments:
        platform_of.setdefault(item.design_point, set()).add(item.platform)
    assert all(len(where) == 1 for where in platform_of.values())
    # Every row of a design point (controls, L/A/N arms, all modes) is assigned together.
    by_dp = {row["design_point"] for row in records if row["run_id"] in expected}
    assert set(platform_of) == by_dp
    assert all(item.estimated_finish_hours >= item.estimated_start_hours for item in assignments)
    for platform in {item.platform for item in assignments}:
        tiers = [item.priority_tier for item in sorted(
            (item for item in assignments if item.platform == platform),
            key=lambda item: (item.estimated_start_hours, item.slot),
        )]
        assert tiers == sorted(tiers)
    summary = write_assignment(assignments, tmp_path / "assignment", records)
    assert summary["rows"] == 274 and summary["overall_makespan_hours"] > 0
    assert summary["unassigned"] == {"rq5_or_tier3_not_assigned": 7, "waiting_for_load_trace": 35}
    assert sum((tmp_path / "assignment" / f"run-list-{p}.txt").read_text().count("\n")
               for p in PLATFORMS) == 274
    with pytest.raises(FileExistsError, match="overwrite"):
        write_assignment(assignments, tmp_path / "assignment")


def test_assignment_platform_hook_needs_a_multiplier(tmp_path: Path) -> None:
    records = _records(tmp_path)
    slots = dict(DEFAULT_SLOTS, cluster_a=8)
    with pytest.raises(ValueError, match="cluster_a"):
        assign(records, slots)
    assignments = assign(records, slots, {"cluster_a": 1.5})
    assert len(assignments) == 274
    points = {}
    for item in assignments:
        points.setdefault(item.design_point, set()).add(item.platform)
    assert all(len(where) == 1 for where in points.values())


def test_assignment_refuses_matrix_without_design_points(tmp_path: Path) -> None:
    records = _records(tmp_path)
    for row in records:
        row["design_point"] = ""
    with pytest.raises(ValueError, match="design_point"):
        assign(records, dict(DEFAULT_SLOTS))


def test_assignment_pins_matching_design_points_without_splitting(tmp_path: Path) -> None:
    records = _records(tmp_path)
    assignments = assign(records, dict(DEFAULT_SLOTS), pins=[("dp_r0p1_*", "cluster_b")])
    platform_of: dict[str, set[str]] = {}
    for item in assignments:
        platform_of.setdefault(item.design_point, set()).add(item.platform)
    assert all(len(where) == 1 for where in platform_of.values())
    assert all(platform_of[key] == {"cluster_b"} for key in platform_of if key.startswith("dp_r0p1_"))


def test_assignment_rejects_invalid_or_conflicting_pins(tmp_path: Path) -> None:
    records = _records(tmp_path)
    with pytest.raises(ValueError, match="unknown pin platform"):
        assign(records, dict(DEFAULT_SLOTS), pins=[("dp_*", "unknown")])
    with pytest.raises(ValueError, match="matched no design point"):
        assign(records, dict(DEFAULT_SLOTS), pins=[("missing_*", "cluster_b")])
    with pytest.raises(ValueError, match="pinned to both"):
        assign(records, dict(DEFAULT_SLOTS), pins=[
            ("dp_r0p1_*", "cluster_b"), ("dp_r0p1_s3_*", "mac_m4_max"),
        ])


def test_assignment_places_long_group_on_many_slot_platform() -> None:
    def row(run_id: str, design_point: str, hours: float) -> dict[str, str]:
        return {
            "run_id": run_id,
            "priority_tier": "1",
            "rqs": "RQ2",
            "availability": "ready_after_pilot",
            "design_point": design_point,
            "estimated_wall_hours_mac_m4_max": str(hours),
        }

    rows = [row(f"long-{i}", "long", 10.0) for i in range(4)]
    rows += [row(f"bulk-{i}", "bulk", 1.0) for i in range(50)]
    assignments = assign(rows, {"fast": 1, "wide": 4}, {"fast": 1.0, "wide": 2.0})
    assert {item.platform for item in assignments if item.design_point == "long"} == {"wide"}
