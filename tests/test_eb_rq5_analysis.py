"""Tests for the EXP-2026-002 RQ5 analysis (H6-H8): block bootstrap, completion metrics, decision rules."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.edgebench.e2e.analysis import (
    SEED,
    apply_holm,
    block_draws,
    block_index,
    boot_percentile,
    boot_rate,
    build_arm,
    cell_row,
    ci,
    false_admission_b,
    h6_tests,
    h6_verdicts,
    h7_verdicts,
    h8_tests,
    h8_verdicts,
    load_part,
    operational_a,
    pair_stats,
    strict_a,
    terminal_status,
)

TRUTH = {"service_type": "count", "locality": "remote_allowed", "quality_floor": "standard", "urgency": "normal"}


def a_row(i: int, ok: bool, T: float = 0.5) -> dict:
    arrival = i * 0.25
    if ok:
        return dict(id=i, arrival=arrival, status="success", predicted=dict(TRUTH), truth=dict(TRUTH), node=0,
                    origin=0, priority=1, terminal=arrival + T, deadline=arrival + 2.0, T=T,
                    service_queue_s=0.0, transfer_s=0.01, service_s=0.05)
    return dict(id=i, arrival=arrival, status="queue_expired", predicted=None, truth=dict(TRUTH), node=None,
                origin=0, priority=None, terminal=arrival + 2.0, deadline=arrival + 2.0, T=2.0)


def arm_from_block_counts(cell: str, model: str, ok_per_block: list[int], T: float = 0.5) -> dict:
    """Part A arm of 30-arrival blocks; block b has its first ok_per_block[b] arrivals completed."""
    rows = [a_row(b * 30 + j, j < k, T) for b, k in enumerate(ok_per_block) for j in range(30)]
    return build_arm("A", cell, model, rows)


def width(lo_hi: tuple) -> float:
    return lo_hi[1] - lo_hi[0]


# --- block bootstrap -------------------------------------------------------------------------------

def test_blocks_are_contiguous_runs_in_arrival_order() -> None:
    assert list(block_index(65)) == [0] * 30 + [1] * 30 + [2] * 5
    # Autocorrelated toy: the first half of the trace completes, the second half does not.
    rows = [a_row(i, i < 150) for i in range(300)]
    arm = build_arm("A", "load_4", "Jev-1.13.0", list(reversed(rows)))  # given out of order
    assert [r["id"] for r in arm["rows"]] == list(range(300))
    row = cell_row(arm, arm, {})
    assert row["strict_completion"] == pytest.approx(0.5)
    # Contiguous blocks keep the autocorrelation: blocks are all-or-nothing, so the CI is wide.
    assert row["strict_ci_high"] - row["strict_ci_low"] > 0.3


def test_block_bootstrap_is_paired_across_models() -> None:
    # Block difficulty varies strongly; the anchor always completes 3 more arrivals per block than l.
    anchor = arm_from_block_counts("load_4", "Jev-1.13.0", [3 + 3 * b for b in range(10)])
    other = arm_from_block_counts("load_4", "DeepSeek-V4.1-Flash", [3 * b for b in range(10)])
    st = pair_stats(other, anchor, {})
    assert st["G_primary"] == pytest.approx(0.1)
    paired = ci(st["G_primary_boot"], st["G_primary"])
    # Unpaired resampling (independent block draws per model) for comparison.
    sup = anchor["supported"]
    c1, c2 = block_draws(10), block_draws(10, seed=SEED + 1)
    unpaired_boot = (boot_rate(anchor["primary"], sup, anchor["block"], c1)
                     - boot_rate(other["primary"], sup, other["block"], c2))
    unpaired = ci(unpaired_boot, st["G_primary"])
    assert width(paired) < 1e-9
    assert width(unpaired) > 0.2


def test_boot_percentile_equals_percentile_of_resampled_arrivals() -> None:
    rng = np.random.default_rng(1)
    values = rng.exponential(size=95)
    block = np.sort(rng.integers(0, 10, size=95))
    counts = block_draws(10, n_resamples=50)
    got = boot_percentile(values, block, counts, 95)
    for r in range(50):
        replicated = np.concatenate([np.repeat(values[block == b], counts[r, b]) for b in range(10)])
        assert got[r] == pytest.approx(np.percentile(replicated, 95))


# --- completion metrics ----------------------------------------------------------------------------

def test_unspecified_vs_normal_urgency_is_operational_not_strict() -> None:
    row = a_row(0, True)
    row["predicted"] = dict(TRUTH, urgency="unspecified")  # maps to the same priority as "normal"
    row["status"] = "wrong_or_late"  # simulator: not an exact joint match
    assert operational_a(row) is True
    assert strict_a(row) is False
    urgent = a_row(1, True)
    urgent["truth"] = dict(TRUTH, urgency="urgent")
    urgent["status"] = "wrong_or_late"  # predicted normal -> priority 1, truth needs 0
    assert operational_a(urgent) is False


def test_part_b_false_admission_counts_only_dispatched_non_ocr_truth() -> None:
    non_ocr = {"service_type": "count", "locality": "site_only", "quality_floor": "standard", "urgency": "normal"}
    ocr = dict(non_ocr, service_type="ocr")
    rows = [
        dict(truth=non_ocr, status="ocr_returned", service_dispatch=1.0),       # false admission
        dict(truth=non_ocr, status="no_feasible_node", service_dispatch=None),  # routed to OCR, never dispatched
        dict(truth=ocr, status="ocr_returned", service_dispatch=2.0),           # OCR truth
    ]
    assert [false_admission_b(r) for r in rows] == [True, False, False]


def test_decision_recorded_without_prediction_is_invalid_decision() -> None:
    assert terminal_status("B", {"id": 1, "status": "decision_recorded", "predicted": None}) == "invalid_decision"
    with pytest.raises(ValueError):
        terminal_status("B", {"id": 2, "status": "decision_recorded", "predicted": dict(TRUTH)})
    score = dict(correct_completion=False, strict_semantic=False, correct_rejection=False, offsite_violation_bytes=0)
    rows = [dict(id=i, arrival=float(i), status="undeployed", predicted=dict(TRUTH), truth=dict(TRUTH), T=0.1,
                 score=score) for i in range(29)]
    rows.append(dict(id=29, arrival=29.0, status="decision_recorded", predicted=None, truth=dict(TRUTH), T=0.3,
                     score=score))
    row = cell_row(build_arm("B", "steady_changing_off", "Jev-1.13.0", rows), None, {})
    assert row["share_invalid_decision"] == pytest.approx(1 / 30)


# --- decision rules --------------------------------------------------------------------------------

def _load_sweep_arms(ok_per_cell: list[int]) -> dict:
    arms = {}
    for cell, k in zip(["load_1", "load_2", "load_4", "load_8", "load_16"], ok_per_cell):
        arms[(cell, "Jev-1.13.0")] = arm_from_block_counts(cell, "Jev-1.13.0", [30] * 10)
        arms[(cell, "DeepSeek-V4.1-Flash")] = arm_from_block_counts(cell, "DeepSeek-V4.1-Flash", [k] * 10)
    return arms


def _h6_verdict(ok_per_cell: list[int]) -> dict:
    tests = h6_tests(_load_sweep_arms(ok_per_cell), {})
    apply_holm(tests)
    return next(v for v in h6_verdicts(tests) if v["contrast"] == "DeepSeek-V4.1-Flash")


def test_h6_slope_sign() -> None:
    widening = _h6_verdict([27, 24, 21, 18, 15])  # G = 0.1 .. 0.5, slope +0.1 per doubling of lambda
    assert widening["verdict"] == "holds"
    narrowing = _h6_verdict([15, 18, 21, 24, 27])
    assert narrowing["verdict"] == "does not hold"


def _h7_test(cell: str, n_shared: int, point: float, resolved: bool) -> dict:
    return dict(hypothesis="H7", part="A", contrast="DeepSeek-V4.1-Flash", cell=cell, n_shared=n_shared,
                point=point, resolved=resolved)


def _h7_deepseek(tests: list[dict]) -> dict:
    return next(v for v in h7_verdicts(tests) if v["contrast"] == "DeepSeek-V4.1-Flash")


def test_h7_excludes_cells_under_30_shared_successes() -> None:
    tests = [_h7_test(f"c{i}", 40, 0.5, True) for i in range(5)]
    tests.append(_h7_test("small", 29, -0.5, True))  # reversed and resolved, but ineligible
    v = _h7_deepseek(tests)
    assert v["n_eligible"] == 5
    assert v["ineligible_cells"] == "A:small"
    assert v["verdict"] == "holds"


def test_h7_needs_at_least_five_eligible_cells() -> None:
    v = _h7_deepseek([_h7_test(f"c{i}", 40, 0.5, True) for i in range(4)])
    assert v["n_eligible"] == 4
    assert v["verdict"] == "not evaluable"


def test_h8_uses_absolute_gaps() -> None:
    # Cache off: Jev ahead by 0.2. Cache on: DeepSeek ahead by 0.2. |gap| is unchanged, so no shrinkage.
    arms = {
        ("reuse_repeated_off", "Jev-1.13.0"): arm_from_block_counts("reuse_repeated_off", "Jev-1.13.0", [27] * 10),
        ("reuse_repeated_off", "DeepSeek-V4.1-Flash"): arm_from_block_counts("reuse_repeated_off", "DeepSeek-V4.1-Flash", [21] * 10),
        ("reuse_repeated_on", "Jev-1.13.0"): arm_from_block_counts("reuse_repeated_on", "Jev-1.13.0", [21] * 10),
        ("reuse_repeated_on", "DeepSeek-V4.1-Flash"): arm_from_block_counts("reuse_repeated_on", "DeepSeek-V4.1-Flash", [27] * 10),
    }
    tests = h8_tests({"A": arms, "B": {}}, {"A": {}, "B": {}})
    apply_holm(tests)
    g = next(t for t in tests if t["contrast"] == "DeepSeek-V4.1-Flash" and t["part"] == "A"
             and t["statistic"] == "did_abs_G")
    assert abs(g["point"]) < 1e-9
    v = next(v for v in h8_verdicts(tests) if v["contrast"] == "DeepSeek-V4.1-Flash" and v["part"] == "A")
    assert v["verdict"] == "does not hold"


# --- coverage --------------------------------------------------------------------------------------

def test_arm_without_integrity_is_a_gap_and_never_read(tmp_path: Path) -> None:
    arm = tmp_path / "load_4" / "seed_1" / "Jev-1.13.0"
    arm.mkdir(parents=True)
    (arm / "outcomes.jsonl").write_text("not json\n")  # would raise if read
    arms, gaps, _ = load_part(tmp_path, "A")
    assert arms == {}
    assert {"part": "A", "cell": "load_4", "model": "Jev-1.13.0",
            "reason": "no integrity.json (missing or still running)"} in gaps
    (arm / "integrity.json").write_text(json.dumps({"complete": True, "arrivals": 30}))
    _, gaps, _ = load_part(tmp_path, "A")
    assert any(g["cell"] == "load_4" and g["model"] == "Jev-1.13.0" and "30 != 300" in g["reason"] for g in gaps)
