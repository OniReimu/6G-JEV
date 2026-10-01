from __future__ import annotations

import copy

import numpy as np
import pytest

from src.ranbench.c2 import stats
from src.ranbench.c2.eligibility import d10_eligibility_flags
from src.ranbench.c2.layout_robustness import layout_flag


def test_d10_block_helper_uses_10s_floor_and_preserves_block_count() -> None:
    result = stats.d10_block_assignments(
        np.array([0.0, 99.0]), 0.0, 100.0,
        {f"arm-{index}": value for index, value in enumerate((1.0, 2.0, 3.0, 4.0, 9.9))},
        10.0, 10,
    )
    assert result["tau_s"] == pytest.approx(9.9)
    assert result["primary"]["block_s"] == pytest.approx(10.0)
    assert result["primary"]["n_blocks"] == 10
    assert result["sensitivity_10s"]["n_blocks"] == 10


def test_d10_block_helper_gives_seven_blocks_for_long_tau() -> None:
    result = stats.d10_block_assignments(
        np.array([2.0, 300.0]), 2.0, 345.3,
        {f"arm-{index}": value for index, value in enumerate((41.0, 42.0, 43.0, 44.0, 48.8))},
        10.0, 34,
    )
    assert result["tau_s"] == pytest.approx(48.8)
    assert result["primary"]["block_s"] == pytest.approx(48.8)
    assert result["primary"]["n_blocks"] == 7


def test_d10_resolution_requires_both_block_analyses() -> None:
    resolved = {"ci_low": 0.01, "ci_high": 0.05, "p_holm": 0.01}
    unresolved = {"ci_low": -0.01, "ci_high": 0.05, "p_holm": 0.01}
    assert stats.resolved_both(resolved, resolved, "p_holm") == (True, False)
    assert stats.resolved_both(resolved, unresolved, "p_holm") == (False, False)

    c1_resolved = {
        "point": 0.06, "ci_low": 0.01, "ci_high": 0.10, "p_raw": 0.01, "pass": True,
    }
    c1_unresolved = c1_resolved | {"ci_low": -0.01, "pass": False}
    c2_pass = {"pass": True}
    flags = d10_eligibility_flags(
        {"C-1": c1_resolved, "C-2": c2_pass},
        {"C-1": c1_unresolved, "C-2": c2_pass},
    )
    assert flags["eligible"] is True
    assert flags["eligible_10s"] is False
    assert flags["eligible_d10"] is False


def test_layout_flag_detects_sign_flip_and_excess_range() -> None:
    sign_flip = layout_flag([0.04, 0.03, -0.01, 0.02])
    assert sign_flip["sign_differs"] is True
    assert sign_flip["not_robust"] is True

    excess_range = layout_flag([0.04, 0.005, 0.06, 0.01])
    assert excess_range["sign_differs"] is False
    assert excess_range["range"] == pytest.approx(0.055)
    assert excess_range["range_exceeds_production"] is True
    assert excess_range["not_robust"] is True


def test_block_assignment_from_stream_does_not_mutate_stream() -> None:
    stream = {
        "meta": {"t0_s": 2.0, "block_s": 10.0, "n_blocks": 10},
        "events": [{"j": 0, "t_issue_s": 2.0}, {"j": 1, "t_issue_s": 50.0}],
    }
    before = copy.deepcopy(stream)
    stats.d10_block_assignments_from_stream(
        stream, 102.0, {f"arm-{index}": 5.0 for index in range(5)})
    assert stream == before


def test_sensitivity_10s_is_the_registered_stream_blocking() -> None:
    # Run end 345.3 s would give floor(343.3/10) = 34 blocks; the frozen stream registers 33.
    t_issue = [2.0, 15.0, 170.0, 331.9, 341.0]
    stream = {
        "meta": {"t0_s": 2.0, "block_s": 10.0, "n_blocks": 33},
        "events": [{"j": j, "t_issue_s": t} for j, t in enumerate(t_issue)],
    }
    result = stats.d10_block_assignments_from_stream(
        stream, 345.3, {f"arm-{index}": 5.0 for index in range(5)})
    sensitivity = result["sensitivity_10s"]
    expected = stats.block_ids(np.array(t_issue), 2.0, 10.0, 33)
    assert sensitivity["n_blocks"] == 33
    assert sensitivity["block_s"] == 10.0
    np.testing.assert_array_equal(sensitivity["blocks"], expected)
    np.testing.assert_array_equal(sensitivity["counts"], stats.draws(33))


def test_sensitivity_rejects_a_registered_block_other_than_10s() -> None:
    stream = {
        "meta": {"t0_s": 2.0, "block_s": 20.0, "n_blocks": 5},
        "events": [{"j": 0, "t_issue_s": 2.0}],
    }
    with pytest.raises(ValueError, match="registered block length"):
        stats.d10_block_assignments_from_stream(
            stream, 102.0, {f"arm-{index}": 5.0 for index in range(5)})


def test_primary_tail_event_goes_to_last_block() -> None:
    # B = 48.8 s, n = 7 blocks ending at 2 + 7 * 48.8 = 343.6 s; the event at 344.0 s is in the tail.
    result = stats.d10_block_assignments(
        np.array([2.0, 100.0, 344.0]), 2.0, 345.3,
        {f"arm-{index}": value for index, value in enumerate((41.0, 42.0, 43.0, 44.0, 48.8))},
        10.0, 34,
    )
    primary = result["primary"]
    assert primary["n_blocks"] == 7
    assert 344.0 > 2.0 + primary["n_blocks"] * primary["block_s"]
    assert primary["blocks"].tolist() == [0, 2, 6]
    assert (primary["blocks"] >= 0).all()


def test_gap_point_estimate_is_blocking_invariant_with_tail_event() -> None:
    t_issue = np.array([2.0, 30.0, 100.0, 200.0, 300.0, 344.0])
    x = np.array([0.5, 0.4, 0.7, 0.2, 0.9, 5.0])
    y = np.array([0.1, 0.3, 0.2, 0.2, 0.1, 1.0])
    result = stats.d10_block_assignments(
        t_issue, 2.0, 345.3,
        {f"arm-{index}": value for index, value in enumerate((41.0, 42.0, 43.0, 44.0, 48.8))},
        10.0, 33,
    )
    primary = result["primary"]
    sensitivity = result["sensitivity_10s"]
    assert primary["blocks"][-1] == primary["n_blocks"] - 1
    g_primary = stats.gap(x, y, primary["blocks"], primary["counts"])
    g_10s = stats.gap(x, y, sensitivity["blocks"], sensitivity["counts"])
    assert g_primary["n"] == g_10s["n"] == len(t_issue)
    assert g_primary["point"] == g_10s["point"] == pytest.approx(float((x - y).mean()))


# ---------------------------------------------------------------- L-arm resolution = hypotheses.json Holm

from src.ranbench.c2.analysis import (  # noqa: E402
    BASE_POINT,
    _dual_gap,
    _dual_stat_columns,
    attach_holm_resolution,
    hypotheses,
)
from src.ranbench.c2.pilot import iot_gap  # noqa: E402

HOSTED = "DeepSeek-V4.1-Flash"
ANCHOR = "Jev-1.13.0"


def _l_row(design_point: str, arrays: dict, context: dict) -> dict:
    """An L-arm gap row as analysis._l_rows builds it (raw p only, before Holm attachment)."""
    aff, aff_10s = _dual_gap(arrays[HOSTED]["aff"], arrays[ANCHOR]["aff"], context)
    net, net_10s = _dual_gap(arrays[HOSTED]["net"], arrays[ANCHOR]["net"], context)
    return {
        "design_point": design_point, "interpreter": HOSTED, "comparator": ANCHOR,
        **_dual_stat_columns("affected_gap", aff, aff_10s, include_p=True),
        **_dual_stat_columns("network_gap", net, net_10s, include_p=True),
    }


def test_l_arm_resolution_is_the_holm_result_hypotheses_reports() -> None:
    # One event per block; gap mean 0.6 has raw p = 0.0384 with CI excluding 0 under both blockings.
    diff = np.array([1.0, -1.0] * 6) + 0.6
    assignment = {"block_s": 10.0, "n_blocks": 12, "blocks": np.arange(12), "counts": stats.draws(12)}
    assignments = {"primary": assignment, "sensitivity_10s": assignment}
    arrays = {
        ANCHOR: {"aff": np.zeros(12), "net": np.zeros(12)},
        HOSTED: {"aff": diff, "net": diff.copy()},
    }
    point = {"assignments": assignments, "arrays": arrays, "eligible_d10": True,
             "eligible_original_5pp_d10": True}
    points = {BASE_POINT: point | {"is_rq2": False}, "dp-rq2": point | {"is_rq2": True}}
    context = {"assignments": assignments}
    rows = [_l_row(BASE_POINT, arrays, context), _l_row("dp-rq2", arrays, context),
            {"design_point": BASE_POINT, "interpreter": ANCHOR, "comparator": None}]
    base_row, rq2_row, anchor_row = rows

    # Raw p passes under both blockings for the base-point H1 contrasts ...
    for prefix in ("affected_gap", "network_gap"):
        for suffix in ("", "_10s"):
            assert base_row[f"{prefix}{suffix}_p_raw"] < stats.ALPHA
            assert base_row[f"{prefix}{suffix}_ci_low"] > 0

    result = hypotheses(points)
    attach_holm_resolution(rows, result["primary_d10"])

    # ... but Holm over the two-test H1 family fails, and the CSV row says what hypotheses.json says.
    h1_tests = {t["metric"]: t for t in result["primary_d10"]["H1"]["result"]["tests"]}
    for prefix, metric in (("affected_gap", "aff"), ("network_gap", "net")):
        test = h1_tests[metric]
        assert test["primary_B"]["p_holm"] >= stats.ALPHA
        assert base_row[f"{prefix}_holm_family"] == "primary_d10:H1"
        assert base_row[f"{prefix}_p_holm"] == test["primary_B"]["p_holm"]
        assert base_row[f"{prefix}_p_holm_10s"] == test["sensitivity_10s"]["p_holm"]
        assert base_row[f"{prefix}_resolved_both"] == bool(test["resolved_pos"] or test["resolved_neg"])
        assert base_row[f"{prefix}_resolved_both"] is False
        assert base_row[f"{prefix}_resolved_B"] is False
        assert base_row[f"{prefix}_resolved_10s"] is False
    assert result["primary_d10"]["H1"]["result"]["holds"][f"{HOSTED} vs {ANCHOR}"] is False

    # At an eligible RQ2 point the affected gap is the only H2 member for this pair (family of one).
    h2_test = next(t for t in result["primary_d10"]["H2"]["result"]["tests"]
                   if t["design_point"] == "dp-rq2")
    assert rq2_row["affected_gap_holm_family"] == "primary_d10:H2"
    assert rq2_row["affected_gap_p_holm"] == h2_test["primary_B"]["p_holm"]
    assert rq2_row["affected_gap_resolved_both"] == bool(h2_test["resolved_pos"] or h2_test["resolved_neg"])
    assert rq2_row["affected_gap_resolved_both"] is True

    # Contrasts outside every Holm family carry no resolution field.
    for key in ("holm_family", "p_holm", "p_holm_10s", "resolved_B", "resolved_10s", "resolved_both"):
        assert rq2_row[f"network_gap_{key}"] is None
        assert anchor_row[f"affected_gap_{key}"] is None
        assert anchor_row[f"network_gap_{key}"] is None


def test_iot_gap_is_reported_and_resolved_under_both_blockings() -> None:
    no_update = np.array([1.0] * 6 + [-0.2] * 6)
    oracle = np.zeros(12)
    primary = {"blocks": np.arange(12), "counts": stats.draws(12)}
    two_blocks = {"blocks": np.array([0] * 6 + [1] * 6), "counts": stats.draws(2)}

    result = iot_gap(no_update, oracle, {"primary": primary, "sensitivity_10s": two_blocks})
    b, ten = result["aff_W2_no_update_minus_oracle"], result["aff_W2_no_update_minus_oracle_10s"]
    assert b["point"] == ten["point"] == pytest.approx(0.4)
    assert b["ci_low"] > 0 and b["p_raw"] < stats.ALPHA
    assert ten["ci_low"] <= 0
    assert result["aff_W2_no_update_minus_oracle_resolved_both_raw_p"] is False

    same = iot_gap(no_update, oracle, {"primary": primary, "sensitivity_10s": primary})
    assert same["aff_W2_no_update_minus_oracle_resolved_both_raw_p"] is True


# ---------------------------------------------------------------- layout check: one binary

import json  # noqa: E402
from pathlib import Path  # noqa: E402

from src.ranbench.c2 import layout_robustness  # noqa: E402

LAYOUT_ARMS = {
    "oracle": ("oracle", "control", ""),
    "no-update": ("no-update", "control", ""),
    **{model: ("L", model, "E") for model in (ANCHOR, *layout_robustness.HOSTED_MODELS)},
}


def _layout_tree(tmp_path: Path, sha_by_root_arm: dict[tuple[int, str], str | None]):
    rows = {name: {"run_id": f"run-{index}"} for index, name in enumerate(LAYOUT_ARMS)}
    roots = []
    for root_index in range(4):
        root = tmp_path / f"root{root_index}"
        for arm, row in rows.items():
            run_dir = root / row["run_id"]
            run_dir.mkdir(parents=True)
            sha = sha_by_root_arm.get((root_index, arm), "a" * 64)
            manifest = {} if sha is None else {"binary_sha256": sha}
            (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        roots.append((f"root{root_index}", root))
    return roots, rows


def test_layout_check_accepts_one_common_binary(tmp_path: Path) -> None:
    roots, rows = _layout_tree(tmp_path, {})
    assert layout_robustness.common_binary_sha256(roots, rows) == "a" * 64


def test_layout_check_rejects_binary_mismatch_across_roots(tmp_path: Path) -> None:
    # Every arm of layout root 2 ran a different binary; each root is internally consistent.
    mismatch = {(2, arm): "b" * 64 for arm in LAYOUT_ARMS}
    roots, rows = _layout_tree(tmp_path, mismatch)
    with pytest.raises(RuntimeError, match="one binary_sha256 across all arms and roots"):
        layout_robustness.common_binary_sha256(roots, rows)


def test_layout_check_rejects_binary_mismatch_across_arms(tmp_path: Path) -> None:
    roots, rows = _layout_tree(tmp_path, {(0, "GLM-5.3-Flash"): "c" * 64})
    with pytest.raises(RuntimeError, match="one binary_sha256 across all arms and roots"):
        layout_robustness.common_binary_sha256(roots, rows)


def test_layout_check_rejects_missing_binary_hash(tmp_path: Path) -> None:
    roots, rows = _layout_tree(tmp_path, {(3, "oracle"): None})
    with pytest.raises(RuntimeError, match="no binary_sha256"):
        layout_robustness.common_binary_sha256(roots, rows)


def test_allow_incomplete_does_not_bypass_binary_check(tmp_path: Path) -> None:
    roots, rows = _layout_tree(tmp_path, {(1, "no-update"): "d" * 64})
    matrix = tmp_path / "matrix.csv"
    lines = ["run_id,design_point,arm,interpreter,mode"] + [
        f"{rows[name]['run_id']},dp,{arm},{interpreter},{mode}"
        for name, (arm, interpreter, mode) in LAYOUT_ARMS.items()
    ]
    matrix.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="one binary_sha256 across all arms and roots"):
        layout_robustness.analyse(
            matrix, "dp", roots[0][1], [root for _, root in roots[1:]], allow_incomplete=True)


def test_d7_and_d7b_counter_lines_are_counted_separately() -> None:
    from src.ranbench.c2.analysis import D7_MARKER, D7B_MARKER
    lines = ["EXP2026003_PATCH_D7 imsi=4 t=1.0", "EXP2026003_PATCH_D7B imsi=5 t=2.0 site=SendTxData lcid=4",
             "EXP2026003_PATCH_D7B imsi=6 t=3.0 site=DoTransmitBufferStatusReport lcid=3"]
    assert sum(bool(D7_MARKER.search(line)) for line in lines) == 1
    assert sum(bool(D7B_MARKER.search(line)) for line in lines) == 2
