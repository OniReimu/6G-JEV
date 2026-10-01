"""Tests for scoring, metrics, and statistical tests."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import binomtest

from src.edgebench.contract import Case, FIELD_SETS
from src.edgebench.interpreters.base import Decision
from src.edgebench.interpreters.oracle import OracleInterpreter
from src.edgebench.scoring import (
    aggregate_metrics,
    bootstrap_ci,
    exact_mcnemar_test,
    holm_adjust,
    score_decision,
)


def test_scoring_oracle_rows_em_one():
    oracle = OracleInterpreter()
    cases = [
        Case(
            case_id=f"c_{i}",
            text=f"Request text {i}",
            fields=FIELD_SETS[4],
            bundle_size=1,
            truth=[{"service_type": "count", "locality": "site_only", "quality_floor": "standard", "urgency": "normal"}],
        )
        for i in range(10)
    ]

    rows = []
    for c in cases:
        decision = oracle.decide(c)
        correct, em, unsafe, spur, miss, unspec_t, spec_t = score_decision(decision, c)
        assert em is True
        assert unsafe is False
        assert spur == 0
        assert miss == 0
        row = {
            "case_id": c.case_id,
            "repeat": 0,
            "em": em,
            "valid": True,
            "correct": correct,
            "unsafe_locality": unsafe,
            "spurious_count": spur,
            "missed_count": miss,
            "unspecified_truth_count": unspec_t,
            "specified_truth_count": spec_t,
            "latency_s": decision.latency_s,
            "cost_usd": decision.cost_usd,
        }
        rows.append(row)

    agg = aggregate_metrics(rows)
    assert agg["em"] == 1.0
    assert agg["valid_rate"] == 1.0
    assert agg["unsafe_rate"] == 0.0
    assert agg["spurious_rate"] == 0.0
    assert agg["missed_rate"] == 0.0


def test_hand_built_unsafe_spurious_missed():
    case = Case(
        case_id="hand_built",
        text="Test text",
        fields=FIELD_SETS[4],
        bundle_size=1,
        truth=[
            {
                "service_type": "ocr",
                "locality": "site_only",
                "quality_floor": "high",
                "urgency": "unspecified",
            }
        ],
    )

    decision = Decision(
        labels=[
            {
                "service_type": "ocr",
                "locality": "remote_allowed",  # unsafe!
                "quality_floor": "unspecified",  # missed!
                "urgency": "urgent",  # spurious!
            }
        ],
        valid=True,
    )

    correct, em, unsafe, spur, miss, unspec_t, spec_t = score_decision(decision, case)
    assert em is False
    assert correct["service_type"] is True
    assert correct["locality"] is False
    assert correct["quality_floor"] is False
    assert correct["urgency"] is False

    assert unsafe is True  # pred remote_allowed, truth site_only
    assert spur == 1  # truth urgency was unspecified, pred specified urgent
    assert miss == 1  # truth quality was high, pred specified unspecified


def test_bootstrap_ci_contains_point_estimate():
    data = [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    point = float(np.mean(data))
    ci_low, ci_high = bootstrap_ci(data, n_resamples=10000, seed=20260924)
    assert ci_low <= point <= ci_high


def test_mcnemar_known_table():
    # Construct discordant pairs: b = 10, c = 2
    # Plus concordant pairs: a = 20, d = 20
    model_em = [True] * 10 + [False] * 2 + [True] * 20 + [False] * 20
    ref_em = [False] * 10 + [True] * 2 + [True] * 20 + [False] * 20

    p_val = exact_mcnemar_test(model_em, ref_em)
    expected_p = float(binomtest(10, n=12, p=0.5).pvalue)
    assert pytest.approx(expected_p, rel=1e-6) == p_val


def test_holm_adjust():
    raw_p = {"test1": 0.01, "test2": 0.04, "test3": 0.03}
    adj = holm_adjust(raw_p)
    # Sorted raw p: 0.01 (m=3 -> 0.03), 0.03 (m=2 -> 0.06), 0.04 (m=1 -> 0.06)
    assert pytest.approx(0.03, rel=1e-5) == adj["test1"]
    assert pytest.approx(0.06, rel=1e-5) == adj["test2"]
    assert pytest.approx(0.06, rel=1e-5) == adj["test3"]
