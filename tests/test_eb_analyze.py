"""Tests for Edgebench analysis pipeline (EXP-2026-001 + Addendum A1)."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
from typing import Any
import pytest
import numpy as np

from src.edgebench.analysis import (
    ALL_EVAL_MODELS,
    ALL_TABLE_MODELS,
    EXPECTED_CONDITIONS,
    EXPECTED_REFERENCE_CELLS,
    MAIN_MODELS,
    REFERENCE_MODEL,
    REFERENCE_MODELS,
    RQ4_REFERENCE_MODELS,
    THE_SIX_MODELS,
    check_coverage,
    compute_availability,
    compute_deadline_violations,
    compute_false_admission_blocking,
    compute_jitter,
    compute_throughput,
    compute_goodput,
    compute_ratio_of_ratios_with_pvalue,
    compute_sla_violation_rate,
    evaluate_hypotheses,
    generate_cells_csv,
    generate_comms_csv,
    generate_flagged_cells_md,
    generate_h5_classifier_reference,
    generate_h5_classifier_reference_md,
    get_model_role,
    integrate_energy,
    invert_bootstrap_pvalue_floored,
    parse_header_tz,
    parse_power_trace,
    strip_model_suffix,
)
from src.edgebench.contract import Case, CORE_FIELDS
from src.edgebench.scoring import bootstrap_ci, exact_mcnemar_test, holm_adjust



def test_energy_integration_synthetic(tmp_path: Path):
    """Energy integration on synthetic trace: constant 100 W for 10 s, 5 decisions -> 200 J/decision; timezone parse checked."""
    trace_file = tmp_path / "power.test.csv"
    trace_file.write_text(
        "# gpu=0 tz=+1000 fields=timestamp,power.draw[W],utilization.gpu[%],memory.used[MiB]\n"
        "2026/09/24 10:00:00.000, 100.0, 0, 1000\n"
        "2026/09/24 10:00:10.000, 100.0, 0, 1000\n",
        encoding="utf-8",
    )

    parsed = parse_power_trace(trace_file)
    tz = parsed["tz"]
    expected_tz = timezone(timedelta(hours=10))
    assert tz == expected_tz

    # Check epoch timestamp matches 2026-09-24 00:00:00 UTC
    expected_t0 = datetime(2026, 9, 24, 0, 0, 0, tzinfo=timezone.utc).timestamp()
    assert pytest.approx(parsed["times"][0], abs=1e-3) == expected_t0
    assert pytest.approx(parsed["times"][1] - parsed["times"][0], abs=1e-3) == 10.0

    # Integrate energy over the 10 s window with 5 decisions
    res = integrate_energy(
        times=parsed["times"],
        powers=parsed["powers"],
        t_start=parsed["times"][0],
        t_end=parsed["times"][1],
        n_decisions=5,
        n_correct=4,
        idle_power=20.0,
    )

    # 100 W * 10 s = 1000 J
    assert pytest.approx(1000.0, rel=1e-5) == res["energy_j"]
    # 1000 J / 5 decisions = 200 J/decision
    assert pytest.approx(200.0, rel=1e-5) == res["energy_j_per_decision"]
    # 1000 J / 4 correct = 250 J/correct decision
    assert pytest.approx(250.0, rel=1e-5) == res["energy_j_per_correct_decision"]
    # Above idle: (1000 J - 20 W * 10 s) / 5 decisions = 800 J / 5 = 160 J/decision
    assert pytest.approx(160.0, rel=1e-5) == res["energy_j_per_decision_above_idle"]


def test_latency_threshold_and_availability():
    """P(latency > tau) and availability on a toy ledger."""
    toy_rows = [
        {"latency_s": 0.04, "error_type": None},
        {"latency_s": 0.08, "error_type": None},
        {"latency_s": 0.15, "error_type": None},
        {"latency_s": 0.25, "error_type": None},
        {"latency_s": 0.60, "error_type": None},
        {"latency_s": 1.20, "error_type": None},
        {"latency_s": 0.03, "error_type": None},
        {"latency_s": 0.09, "error_type": None},
        {"latency_s": None, "error_type": "timeout"},
        {"latency_s": None, "error_type": "http_error"},
    ]

    # Availability: 1 - 2/10 = 0.80
    avail = compute_availability(toy_rows)
    assert pytest.approx(0.80, rel=1e-5) == avail

    # P(latency > tau) over the 8 valid latency calls
    lats = [r["latency_s"] for r in toy_rows if r["latency_s"] is not None]
    res = compute_deadline_violations(lats, thresholds=(0.05, 0.10, 0.20, 0.50, 1.0))

    # Values > 0.10: 0.15, 0.25, 0.60, 1.20 -> 4 / 8 = 0.50
    assert pytest.approx(0.50, rel=1e-5) == res[0.10]["p"]
    ci_low, ci_high = res[0.10]["ci95"]
    assert ci_low <= 0.50 <= ci_high

    # Values > 1.00: 1.20 -> 1 / 8 = 0.125
    assert pytest.approx(0.125, rel=1e-5) == res[1.0]["p"]


def test_false_admission_false_blocking():
    """False-admission/false-blocking on 4 toy RQ4 rows."""
    cases_map = {
        "c1": Case(
            case_id="c1",
            text="case 1",
            truth=[{"service_type": "unsupported", "locality": "unspecified", "quality_floor": "unspecified", "urgency": "unspecified"}],
        ),
        "c2": Case(
            case_id="c2",
            text="case 2",
            truth=[{"service_type": "unsupported", "locality": "unspecified", "quality_floor": "unspecified", "urgency": "unspecified"}],
        ),
        "c3": Case(
            case_id="c3",
            text="case 3",
            truth=[{"service_type": "count", "locality": "unspecified", "quality_floor": "unspecified", "urgency": "unspecified"}],
        ),
        "c4": Case(
            case_id="c4",
            text="case 4",
            truth=[{"service_type": "detection", "locality": "unspecified", "quality_floor": "unspecified", "urgency": "unspecified"}],
        ),
    }

    toy_rows = [
        # c1: truth unsupported, pred ocr (service) -> False admission!
        {"case_id": "c1", "labels": [{"service_type": "ocr"}]},
        # c2: truth unsupported, pred unsupported -> Correct rejection
        {"case_id": "c2", "labels": [{"service_type": "unsupported"}]},
        # c3: truth count (service), pred unsupported -> False blocking!
        {"case_id": "c3", "labels": [{"service_type": "unsupported"}]},
        # c4: truth detection (service), pred detection (service) -> Correct admission
        {"case_id": "c4", "labels": [{"service_type": "detection"}]},
    ]

    rates = compute_false_admission_blocking(toy_rows, cases_map=cases_map)
    # Unsupported targets: 2 (c1, c2). False admissions: 1 (c1). Rate = 1/2 = 0.50
    assert pytest.approx(0.50, rel=1e-5) == rates["false_admission_rate"]
    # Service targets: 2 (c3, c4). False blockings: 1 (c3). Rate = 1/2 = 0.50
    assert pytest.approx(0.50, rel=1e-5) == rates["false_blocking_rate"]


def test_model_name_suffix_stripping():
    """Model-name suffix stripping."""
    assert strip_model_suffix("SemIf-Qwen3.5-4B@cuda") == "SemIf-Qwen3.5-4B"
    assert strip_model_suffix("Laya@cuda") == "Laya"
    assert strip_model_suffix("Jev-1.13.0") == "Jev-1.13.0"
    assert strip_model_suffix("Qwen3.5-4B-JSON@device_0") == "Qwen3.5-4B-JSON"


def test_coverage_check_exits_on_missing_cell():
    """Coverage check exits non-zero (raises ValueError) on a missing cell."""
    # Complete dummy dict with 1 missing cell
    runs_by_cell = {("ModelA", "RQ1a", "base"): [{"repeat": 0}] * 300}
    expected_models = ["ModelA"]

    with pytest.raises(ValueError, match="Coverage check failed"):
        check_coverage(runs_by_cell, models=expected_models, allow_missing=False)

    # When allow_missing=True, it warns and does not raise
    with pytest.warns(UserWarning, match="Coverage check failed"):
        coverage, missing, matrix = check_coverage(
            runs_by_cell, models=expected_models, allow_missing=True
        )
        assert len(missing) > 0


def test_h1_gating_excludes_low_valid_rate():
    """H1 gating excludes a level with valid rate 0.85 (< 0.90)."""
    # Create 100 rows where model has 85 valid (valid rate = 0.85)
    rows_cand_base = [
        {"case_id": f"c_{i}", "valid": (i < 85), "latency_s": 0.5, "repeat": 0}
        for i in range(100)
    ]
    rows_ref_base = [
        {"case_id": f"c_{i}", "valid": True, "latency_s": 0.1, "repeat": 0}
        for i in range(100)
    ]

    runs_by_cell = {
        ("ModelA", "RQ1a", "base"): rows_cand_base,
        ("Jev-1.13.0", "RQ1a", "base"): rows_ref_base,
    }

    # Evaluate H1 on this cell
    cand_valid = float(np.mean([r["valid"] for r in rows_cand_base]))
    assert cand_valid == 0.85
    ref_valid = float(np.mean([r["valid"] for r in rows_ref_base]))
    assert ref_valid == 1.0

    # Rule: candidate and ref must both have valid rate >= 0.90
    is_gated = (cand_valid < 0.90) or (ref_valid < 0.90)
    assert is_gated is True


def test_holm_applied_within_one_family_only():
    """Holm applied within one family only (two families do not mix)."""
    # Family 1 (latency contrasts): 2 tests with p = [0.01, 0.04]
    fam1_raw = {"lat1": 0.01, "lat2": 0.04}
    # Family 2 (ratio tests): 1 test with p = [0.02]
    fam2_raw = {"ratio1": 0.02}

    # If mixed across all 3 tests (m=3):
    # Sorted: 0.01 (m=3 -> 0.03), 0.02 (m=2 -> 0.04), 0.04 (m=1 -> 0.04)
    # But when kept within family:
    fam1_adj = holm_adjust(fam1_raw)
    fam2_adj = holm_adjust(fam2_raw)

    # Family 1 has m=2:
    # 0.01 * 2 = 0.02 (not 0.03!)
    assert pytest.approx(0.02, rel=1e-5) == fam1_adj["lat1"]
    # 0.04 * 1 = 0.04
    assert pytest.approx(0.04, rel=1e-5) == fam1_adj["lat2"]

    # Family 2 has m=1:
    assert pytest.approx(0.02, rel=1e-5) == fam2_adj["ratio1"]


def test_toy_paired_em_contrast_known_sign():
    """A toy paired EM contrast whose sign is known."""
    # Model 1 has EM = 0.9 (9 correct out of 10)
    # Model 2 has EM = 0.2 (2 correct out of 10)
    # Pairs 0..1: both correct
    # Pairs 2..8: M1 correct, M2 wrong (b = 7)
    # Pair 9: both wrong (c = 0)
    m1_em = [True] * 9 + [False]
    m2_em = [True] * 2 + [False] * 8

    diffs = np.asarray(m1_em, dtype=float) - np.asarray(m2_em, dtype=float)
    mean_diff = float(np.mean(diffs))
    assert mean_diff == pytest.approx(0.70, rel=1e-5)

    ci_low, ci_high = bootstrap_ci(diffs, seed=20260924, n_resamples=10000)
    assert ci_low > 0.0  # CI strictly positive

    mcnemar_p = exact_mcnemar_test(m1_em, m2_em)
    # Exact binomial p for 7 discordant pairs, 7 successes: 2 * (0.5)^7 = 0.015625
    assert pytest.approx(0.015625, rel=1e-5) == mcnemar_p
    assert mcnemar_p < 0.05


def test_false_admission_false_blocking_exact_rates():
    """Exact false-admission and false-blocking rates on a toy set with both errors present (kills Mutant 1)."""
    cases_map = {
        f"c{i}": Case(
            case_id=f"c{i}",
            text=f"case {i}",
            truth=[
                {
                    "service_type": "unsupported" if i <= 3 else "count",
                    "locality": "unspecified",
                    "quality_floor": "unspecified",
                    "urgency": "unspecified",
                }
            ],
        )
        for i in range(1, 8)
    }

    # 3 unsupported cases (c1, c2, c3):
    # c1: pred ocr (service) -> false admission
    # c2: pred detection (service) -> false admission
    # c3: pred unsupported -> correct rejection
    # -> False admission rate = 2 / 3.
    # (Under Mutant 1 which tests `pred_svc == "unsupported"`, only c3 is counted, giving 1/3 != 2/3)
    #
    # 4 service cases (c4, c5, c6, c7):
    # c4: pred unsupported -> false blocking
    # c5: pred count -> correct admission
    # c6: pred count -> correct admission
    # c7: pred count -> correct admission
    # -> False blocking rate = 1 / 4 = 0.25.
    rows = [
        {"case_id": "c1", "labels": [{"service_type": "ocr"}]},
        {"case_id": "c2", "labels": [{"service_type": "detection"}]},
        {"case_id": "c3", "labels": [{"service_type": "unsupported"}]},
        {"case_id": "c4", "labels": [{"service_type": "unsupported"}]},
        {"case_id": "c5", "labels": [{"service_type": "count"}]},
        {"case_id": "c6", "labels": [{"service_type": "count"}]},
        {"case_id": "c7", "labels": [{"service_type": "count"}]},
    ]

    rates = compute_false_admission_blocking(rows, cases_map=cases_map)
    assert pytest.approx(2 / 3, rel=1e-5) == rates["false_admission_rate"]
    assert pytest.approx(1 / 4, rel=1e-5) == rates["false_blocking_rate"]


def test_h1a_gating_real_evaluation_path():
    """H1(a) gating through real evaluation path: 0.85 valid is gated out, 0.95 valid is kept (kills Mutant 2)."""
    # 100 cases per level for DeepSeek-V4.1-Flash and Jev-1.13.0
    # Level 'base': DeepSeek has 85 valid (85% valid -> gated under 0.90 gate)
    # Level 'pad_512': DeepSeek has 95 valid (95% valid -> kept under 0.90 gate)
    # Jev-1.13.0 has 100% valid in both.
    runs_by_cell: dict[tuple[str, str, str], list[dict[str, Any]]] = {}

    runs_by_cell[("DeepSeek-V4.1-Flash", "RQ1a", "base")] = [
        {"case_id": f"c_{i:04d}", "valid": (i < 85), "latency_s": 0.5, "repeat": 0}
        for i in range(100)
    ]
    runs_by_cell[("Jev-1.13.0", "RQ1a", "base")] = [
        {"case_id": f"c_{i:04d}", "valid": True, "latency_s": 0.1, "repeat": 0}
        for i in range(100)
    ]

    runs_by_cell[("DeepSeek-V4.1-Flash", "RQ1a", "pad_512")] = [
        {"case_id": f"c_{i:04d}", "valid": (i < 95), "latency_s": 0.6, "repeat": 0}
        for i in range(100)
    ]
    runs_by_cell[("Jev-1.13.0", "RQ1a", "pad_512")] = [
        {"case_id": f"c_{i:04d}", "valid": True, "latency_s": 0.12, "repeat": 0}
        for i in range(100)
    ]

    # Run real evaluation path
    rows, md = evaluate_hypotheses(runs_by_cell, cases_by_rq_cond={}, seed=20260924)

    h1a_base = next(
        c
        for c in rows
        if c["hypothesis"] == "H1(a)"
        and c["model"] == "DeepSeek-V4.1-Flash"
        and c["level"] == "base"
    )
    h1a_pad = next(
        c
        for c in rows
        if c["hypothesis"] == "H1(a)"
        and c["model"] == "DeepSeek-V4.1-Flash"
        and c["level"] == "pad_512"
    )

    # Under genuine 0.90 gate: 0.85 is gated, 0.95 is kept.
    # Under Mutant 2 (gate 0.80): 0.85 would NOT be gated (gated == False) and this assertion fails!
    assert h1a_base["gated"] is True
    assert h1a_pad["gated"] is False


def test_low_validity_flag_threshold_and_propagation():
    """Verify flag_valid_lt_50 is set when valid_rate < 0.50, unset >= 0.50, and propagates to hypotheses and markdown."""
    def _make_row(
        i: int,
        valid: bool,
        latency: float = 0.2,
        err: str | None = None,
        raw: str | None = None,
    ) -> dict[str, Any]:
        return {
            "case_id": f"c_{i:04d}",
            "valid": valid,
            "latency_s": latency,
            "repeat": 0,
            "unsafe_locality": False,
            "spurious_count": 0,
            "missed_count": 0,
            "unspecified_truth_count": 0,
            "specified_truth_count": 1,
            "labels": [{"service_type": "count"}],
            "error_type": err,
            "raw_response": raw,
        }

    runs_by_cell = {
        ("SemIf-Qwen3.5-4B", "RQ4", "churn25"): [
            _make_row(i, i < 49, 0.2, "HTTP_400", "bad options")
            for i in range(100)
        ],
        ("SemIf-Qwen3.5-4B", "RQ4", "churn50"): [
            _make_row(i, i < 50, 0.2)
            for i in range(100)
        ],
        ("Jev-1.13.0", "RQ4", "churn25"): [
            _make_row(i, i < 51, 0.05)
            for i in range(100)
        ],
    }

    # 1. Check cells.csv
    cases_dummy = {"RQ4": {"churn25": {}, "churn50": {}}}
    cells_rows, cells_csv = generate_cells_csv(
        runs_by_cell, cases_dummy, models=["SemIf-Qwen3.5-4B", "Jev-1.13.0"], seed=20260924
    )
    r_c25 = next(
        r for r in cells_rows if r["model"] == "SemIf-Qwen3.5-4B" and r["condition"] == "churn25"
    )
    r_c50 = next(
        r for r in cells_rows if r["model"] == "SemIf-Qwen3.5-4B" and r["condition"] == "churn50"
    )
    r_jev = next(
        r for r in cells_rows if r["model"] == "Jev-1.13.0" and r["condition"] == "churn25"
    )
    assert r_c25["valid_rate"] == 0.49 and r_c25["flag_valid_lt_50"] is True
    assert r_c50["valid_rate"] == 0.50 and r_c50["flag_valid_lt_50"] is False
    assert r_jev["valid_rate"] == 0.51 and r_jev["flag_valid_lt_50"] is False

    # 2. Check comms.csv
    comms_rows, comms_csv = generate_comms_csv(
        runs_by_cell, cases_dummy, models=["SemIf-Qwen3.5-4B", "Jev-1.13.0"], seed=20260924
    )
    cr_c25 = next(
        r for r in comms_rows if r["model"] == "SemIf-Qwen3.5-4B" and r["condition"] == "churn25"
    )
    cr_c50 = next(
        r for r in comms_rows if r["model"] == "SemIf-Qwen3.5-4B" and r["condition"] == "churn50"
    )
    assert cr_c25["flag_valid_lt_50"] is True
    assert cr_c50["flag_valid_lt_50"] is False

    # 3. Check hypotheses.csv and hypotheses.md
    hyp_rows, hyp_md = evaluate_hypotheses(
        runs_by_cell, cases_by_rq_cond=cases_dummy, seed=20260924
    )
    h5_c25 = next(
        c
        for c in hyp_rows
        if c["hypothesis"] == "H5"
        and c["model"] == "SemIf-Qwen3.5-4B"
        and c["level"] == "churn25"
    )
    h5_c50 = next(
        c
        for c in hyp_rows
        if c["hypothesis"] == "H5"
        and c["model"] == "SemIf-Qwen3.5-4B"
        and c["level"] == "churn50"
    )
    assert "(SemIf-Qwen3.5-4B, churn25)" in h5_c25["flag_low_valid"]
    assert h5_c50["flag_low_valid"] == ""
    # Verdict suffix in markdown
    assert "(flagged: valid <50%)" in hyp_md
    # Flagged cells section
    assert "## Flagged cells" in hyp_md
    assert "SemIf-Qwen3.5-4B" in hyp_md
    assert "49.0%" in hyp_md


def test_bootstrap_pvalue_floor_applied_only_to_bootstrap():
    """Verify bootstrap p-values are floored at 1/(B+1) (never 0), while exact p-values are unfloored."""
    # 1. invert_bootstrap_pvalue_floored
    samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    B = 10000
    p_floor_val = 1.0 / (B + 1)
    p = invert_bootstrap_pvalue_floored(samples, null_value=0.0, n_resamples=B)
    assert p == pytest.approx(p_floor_val, rel=1e-5)
    assert p > 0.0

    # 2. compute_ratio_of_ratios_with_pvalue
    m1_a = [10.0] * 20
    m1_b = [1.0] * 20
    m2_a = [1.0] * 20
    m2_b = [10.0] * 20
    pt, ci, raw_p, _ = compute_ratio_of_ratios_with_pvalue(
        m1_a, m1_b, m2_a, m2_b, seed=20260924, n_resamples=10000, null_value=1.0
    )
    assert raw_p == pytest.approx(p_floor_val, rel=1e-5)
    assert raw_p > 0.0

    # 3. Exact tests: exact_mcnemar_test returns exact p < 1e-4 without floor
    a = [True] * 20
    b = [False] * 20
    mcnemar_p = exact_mcnemar_test(a, b)
    assert mcnemar_p == pytest.approx(2.0 * (0.5 ** 20), rel=1e-5)
    assert mcnemar_p < p_floor_val


def test_model_role_classification():
    """Verify role column mapping: main for the six, control for Oracle, reference for all others."""
    for m in THE_SIX_MODELS:
        assert get_model_role(m) == "main"

    assert get_model_role("Oracle") == "control"

    assert get_model_role("Qwen3.5-4B-JSON") == "reference"
    assert get_model_role("Rule") == "reference"
    assert get_model_role("MiniLM-Reranker") == "reference"
    assert get_model_role("DistilBERT-Clf-All") == "reference"
    assert get_model_role("DistilBERT-Clf-Frozen") == "reference"
    assert get_model_role("DistilBERT-Clf-Retrained") == "reference"


def test_h5_reranker_family_size_16():
    """Verify MiniLM-Reranker enters H5 family, giving exactly 16 contrasts, with valid verdict and no placeholder."""
    # 8 interpreters evaluated in H5: 6 main models + Qwen3.5-4B-JSON + MiniLM-Reranker
    h5_models = THE_SIX_MODELS + [REFERENCE_MODEL, "MiniLM-Reranker"]
    assert len(h5_models) == 8

    # Build synthetic rows for churn25 and churn50
    runs_by_cell: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for m in h5_models:
        for cond in ["churn25", "churn50"]:
            rows = []
            for i in range(20):
                # 10 seen cases, 10 unseen cases
                is_seen = (i < 10)
                rows.append({
                    "case_id": f"c_{cond}_{i:03d}",
                    "valid": True,
                    "repeat": 0,
                    "latency_s": 0.05,
                    "meta": {"seen": is_seen},
                    "correct": {"service_type": True},
                    "labels": [{"service_type": "detection"}],
                })
            runs_by_cell[(m, "RQ4", cond)] = rows

    hyp_rows, hyp_md = evaluate_hypotheses(runs_by_cell, cases_by_rq_cond={}, seed=20260924)

    h5_rows = [r for r in hyp_rows if r["hypothesis"] == "H5"]
    # 8 models * 2 churn levels = 16 contrasts
    assert len(h5_rows) == 16

    # Verify all 16 belong to the catalog_gap family
    assert all(r["family"] == "catalog_gap" for r in h5_rows)
    assert all(r["null_val"] == 0.10 for r in h5_rows)
    assert all(r["direction"] == "upper_bound_lt" for r in h5_rows)

    # MiniLM-Reranker rows specifically
    reranker_rows = [r for r in h5_rows if r["model"] == "MiniLM-Reranker"]
    assert len(reranker_rows) == 2
    assert {r["level"] for r in reranker_rows} == {"churn25", "churn50"}
    for r in reranker_rows:
        assert r["verdict"] in ("resolved", "not resolved")
        assert "upper_ci95=" in r["notes"]
        assert isinstance(r["estimate"], float)
        assert isinstance(r["raw_p"], float)
        assert isinstance(r["holm_p"], float)

    # Check markdown table
    assert "reranker not run" not in hyp_md
    assert "MiniLM-Reranker seen - unseen gap @ churn25" in hyp_md
    assert "MiniLM-Reranker seen - unseen gap @ churn50" in hyp_md


def test_classifier_rows_never_enter_hypothesis_family():
    """Verify that classifier references (and Rule / Oracle) never enter any hypothesis family (H1-H5)."""
    classifiers_and_controls = [
        "DistilBERT-Clf-All",
        "DistilBERT-Clf-Frozen",
        "DistilBERT-Clf-Retrained",
        "Rule",
        "Oracle",
    ]
    runs_by_cell: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for m in classifiers_and_controls:
        for rq in ["RQ1a", "RQ4"]:
            for cond in ["base", "churn25", "churn50"]:
                rows = [
                    {
                        "case_id": f"c_{i:03d}",
                        "valid": True,
                        "repeat": 0,
                        "latency_s": 0.05,
                        "meta": {"seen": (i < 10)},
                        "correct": {"service_type": True, "locality": True, "quality_floor": True, "urgency": True},
                        "em": True,
                        "labels": [{"service_type": "detection"}],
                    }
                    for i in range(20)
                ]
                runs_by_cell[(m, rq, cond)] = rows

    # Also add Jev-1.13.0 and MiniLM-Reranker for H5
    for m in ["Jev-1.13.0", "MiniLM-Reranker"]:
        for cond in ["churn25", "churn50"]:
            rows = [
                {
                    "case_id": f"c_{i:03d}",
                    "valid": True,
                    "repeat": 0,
                    "latency_s": 0.05,
                    "meta": {"seen": (i < 10)},
                    "correct": {"service_type": True},
                    "labels": [{"service_type": "detection"}],
                }
                for i in range(20)
            ]
            runs_by_cell[(m, "RQ4", cond)] = rows

    hyp_rows, hyp_md = evaluate_hypotheses(runs_by_cell, cases_by_rq_cond={}, seed=20260924)

    # Confirm that no classifier, Rule, or Oracle model appears in any hypothesis contrast
    models_in_contrasts = {r["model"] for r in hyp_rows}
    for forbidden in classifiers_and_controls:
        assert forbidden not in models_in_contrasts, f"{forbidden} unexpectedly entered hypothesis family"

    allowed_models = set(THE_SIX_MODELS) | {REFERENCE_MODEL, "MiniLM-Reranker", "The Six"}
    assert models_in_contrasts.issubset(allowed_models)


def test_cells_and_comms_role_column_and_rq4_reference_em_empty():
    """Verify role column values, Oracle exclusion, and empty EM/service metrics for RQ4 references."""
    cases_map = {
        f"c_{i:03d}": Case(
            case_id=f"c_{i:03d}",
            text=f"case {i}",
            truth=[{
                "service_type": "unsupported" if i < 5 else "detection",
                "locality": "unspecified",
                "quality_floor": "unspecified",
                "urgency": "unspecified",
            }],
            meta={"seen": (i >= 5 and i < 15)},
        )
        for i in range(20)
    }
    cases_dummy = {"RQ4": {"churn25": cases_map}, "RQ1a": {"base": cases_map}}

    def _make_rows(is_rq4_ref: bool = False):
        rows = []
        for i in range(20):
            rows.append({
                "case_id": f"c_{i:03d}",
                "valid": True,
                "repeat": 0,
                "latency_s": 0.05,
                "em": True,
                "macro_field_acc": 1.0,
                "unsafe_locality": False,
                "spurious_count": 0,
                "missed_count": 0,
                "unspecified_truth_count": 0,
                "specified_truth_count": 1,
                "correct": {"service_type": True, "locality": True, "quality_floor": True, "urgency": True},
                "labels": [{"service_type": "unsupported" if i < 5 else "detection"}],
                "error_type": None,
            })
        return rows

    runs_by_cell = {
        ("Jev-1.13.0", "RQ4", "churn25"): _make_rows(is_rq4_ref=False),
        ("Rule", "RQ1a", "base"): _make_rows(is_rq4_ref=False),
        ("MiniLM-Reranker", "RQ4", "churn25"): _make_rows(is_rq4_ref=True),
        ("DistilBERT-Clf-Frozen", "RQ4", "churn25"): _make_rows(is_rq4_ref=True),
        ("Oracle", "RQ4", "churn25"): _make_rows(is_rq4_ref=False),
    }

    test_models = ["Jev-1.13.0", "Rule", "MiniLM-Reranker", "DistilBERT-Clf-Frozen", "Oracle"]

    # 1. Test cells.csv
    cells_rows, cells_csv = generate_cells_csv(
        runs_by_cell, cases_dummy, models=test_models, seed=20260924
    )

    models_in_cells = {r["model"] for r in cells_rows}
    assert "Oracle" not in models_in_cells
    assert all(r["role"] != "control" for r in cells_rows)

    jev_cell = next(r for r in cells_rows if r["model"] == "Jev-1.13.0")
    rule_cell = next(r for r in cells_rows if r["model"] == "Rule")
    minilm_cell = next(r for r in cells_rows if r["model"] == "MiniLM-Reranker")
    clf_cell = next(r for r in cells_rows if r["model"] == "DistilBERT-Clf-Frozen")

    assert jev_cell["role"] == "main"
    assert rule_cell["role"] == "reference"
    assert minilm_cell["role"] == "reference"
    assert clf_cell["role"] == "reference"

    # Main model (Jev-1.13.0) and Rule have EM and macro field acc filled
    assert jev_cell["em"] != ""
    assert jev_cell["macro_field_acc"] != ""
    assert rule_cell["em"] != ""
    assert rule_cell["macro_field_acc"] != ""

    # RQ4 references have EM / macro field acc / unsafe / spurious / missed left empty
    for rq4_ref in (minilm_cell, clf_cell):
        assert rq4_ref["em"] == ""
        assert rq4_ref["em_ci_low"] == ""
        assert rq4_ref["em_ci_high"] == ""
        assert rq4_ref["macro_field_acc"] == ""
        assert rq4_ref["macro_field_acc_ci_low"] == ""
        assert rq4_ref["macro_field_acc_ci_high"] == ""
        assert rq4_ref["unsafe_rate"] == ""
        assert rq4_ref["unsafe_rate_ci_low"] == ""
        assert rq4_ref["unsafe_rate_ci_high"] == ""
        assert rq4_ref["spurious_rate"] == ""
        assert rq4_ref["spurious_rate_ci_low"] == ""
        assert rq4_ref["spurious_rate_ci_high"] == ""
        assert rq4_ref["missed_rate"] == ""
        assert rq4_ref["missed_rate_ci_low"] == ""
        assert rq4_ref["missed_rate_ci_high"] == ""

        # But service metrics and latency are filled
        assert rq4_ref["seen_top1"] != ""
        assert rq4_ref["unseen_top1"] != ""
        assert rq4_ref["unsupported_f1"] != ""
        assert rq4_ref["latency_p50"] != ""

    # 2. Test comms.csv
    comms_rows, comms_csv = generate_comms_csv(
        runs_by_cell, cases_dummy, models=test_models, seed=20260924
    )

    models_in_comms = {r["model"] for r in comms_rows}
    assert "Oracle" not in models_in_comms
    assert all(r["role"] != "control" for r in comms_rows)

    jev_comm = next(r for r in comms_rows if r["model"] == "Jev-1.13.0")
    minilm_comm = next(r for r in comms_rows if r["model"] == "MiniLM-Reranker")
    clf_comm = next(r for r in comms_rows if r["model"] == "DistilBERT-Clf-Frozen")

    assert jev_comm["goodput_correct_per_s"] != ""
    assert jev_comm["sla_violation_rate"] != ""

    # RQ4 references have goodput and SLA violation rate empty
    for rq4_comm in (minilm_comm, clf_comm):
        assert rq4_comm["goodput_correct_per_s"] == ""
        assert rq4_comm["sla_violation_rate"] == ""
        # Jitter, latency thresholds, throughput, false admission / blocking filled
        assert rq4_comm["jitter_iqr_s"] != ""
        assert rq4_comm["throughput_decisions_per_s"] != ""
        assert rq4_comm["rq4_false_admission_rate"] != ""
        assert rq4_comm["rq4_false_blocking_rate"] != ""


def test_coverage_check_references():
    """Verify check_coverage enforces n=300 for all expected reference cells when check_references=True."""
    complete_ref_runs: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for m, cells in EXPECTED_REFERENCE_CELLS.items():
        for rq, cond in cells:
            complete_ref_runs[(m, rq, cond)] = [{"repeat": 0}] * 300

    complete_runs = dict(complete_ref_runs)
    for m in ALL_EVAL_MODELS:
        for rq, conds in EXPECTED_CONDITIONS.items():
            for cond in conds:
                complete_runs[(m, rq, cond)] = [{"repeat": 0}] * 300

    # 1. Complete dataset passes
    cov, missing, matrix = check_coverage(complete_runs, models=ALL_EVAL_MODELS, check_references=True)
    assert len(missing) == 0

    # 2. Missing reference cell fails
    incomplete_missing = dict(complete_runs)
    del incomplete_missing[("Rule", "RQ1a", "base")]
    with pytest.raises(ValueError, match="Coverage check failed"):
        check_coverage(incomplete_missing, models=ALL_EVAL_MODELS, check_references=True)

    # 3. Reference cell with count != 300 fails
    incomplete_count = dict(complete_runs)
    incomplete_count[("MiniLM-Reranker", "RQ4", "churn25")] = [{"repeat": 0}] * 299
    with pytest.raises(ValueError, match="Coverage check failed"):
        check_coverage(incomplete_count, models=ALL_EVAL_MODELS, check_references=True)


def test_h5_classifier_reference_generation(tmp_path: Path):
    """Verify generate_h5_classifier_reference reads training.json adaptation cost and outputs table."""
    runs_dir = tmp_path / "runs" / "EXP-2026-001"
    models_dir = runs_dir / "_reference" / "_models"
    models_dir.mkdir(parents=True)

    (models_dir / "clf_retrained_25").mkdir()
    (models_dir / "clf_retrained_25" / "training.json").write_text(
        json.dumps({"adaptation_examples_count": 76, "wall_time_s": 12.815}), encoding="utf-8"
    )

    (models_dir / "clf_retrained_50").mkdir()
    (models_dir / "clf_retrained_50" / "training.json").write_text(
        json.dumps({"adaptation_examples_count": 92, "wall_time_s": 10.022}), encoding="utf-8"
    )

    (models_dir / "clf_frozen").mkdir()
    (models_dir / "clf_frozen" / "training.json").write_text(
        json.dumps({"adaptation_examples_count": 0, "wall_time_s": None}), encoding="utf-8"
    )

    runs_by_cell: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for m in ["DistilBERT-Clf-Frozen", "DistilBERT-Clf-Retrained"]:
        for cond in ["churn25", "churn50"]:
            rows = []
            for i in range(20):
                is_seen = (i < 10)
                rows.append({
                    "case_id": f"c_{cond}_{i:03d}",
                    "valid": True,
                    "repeat": 0,
                    "meta": {"seen": is_seen},
                    "correct": {"service_type": True if is_seen else False},
                    "labels": [{"service_type": "detection"}],
                })
            runs_by_cell[(m, "RQ4", cond)] = rows

    clf_rows, clf_csv = generate_h5_classifier_reference(runs_by_cell, cases_by_rq_cond={}, runs_dir=runs_dir)

    # 2 churn levels * 2 models = 4 rows
    assert len(clf_rows) == 4

    r25_retrained = next(r for r in clf_rows if r["churn_level"] == "churn25" and r["model"] == "DistilBERT-Clf-Retrained")
    assert r25_retrained["added_labelled_examples"] == 76
    assert pytest.approx(r25_retrained["training_wall_time_s"], rel=1e-3) == 12.815

    r25_frozen = next(r for r in clf_rows if r["churn_level"] == "churn25" and r["model"] == "DistilBERT-Clf-Frozen")
    assert r25_frozen["added_labelled_examples"] == 0
    assert r25_frozen["training_wall_time_s"] == ""

    # Check markdown table generation
    md = generate_h5_classifier_reference_md(clf_rows)
    assert "DistilBERT-Clf-Frozen" in md
    assert "DistilBERT-Clf-Retrained" in md
    assert f"{r25_retrained['training_wall_time_s']:.2f}" in md



def test_summary_notes_are_computed_from_verdicts():
    from src.edgebench.analysis import _unresolved_note
    assert _unresolved_note([{"contrast": "a", "verdict": "resolved"}]) == "all resolved"
    assert _unresolved_note([{"contrast": "a", "verdict": "resolved"}, {"contrast": "b", "verdict": "not resolved"}]) == "not resolved: b"
    assert _unresolved_note([{"contrast": "c", "verdict": "resolved (flagged: valid <50%)"}]) == "all resolved"


def test_report_md_has_no_hardcoded_result_claims():
    import inspect
    from src.edgebench import analysis
    src = inspect.getsource(analysis.generate_report_md) + inspect.getsource(analysis.evaluate_hypotheses)
    for claim in ("super-linearly", "16/16", "p < 1e-8", "All 4 contrasts", "All 8 conditions", "2026-09-24 |"):
        assert claim not in src, claim
