from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.ranbench.c2 import stats
from src.ranbench.c2.analysis import (
    BASE_POINT,
    _verify,
    an_decomposition,
    discover_run_dirs,
    hypotheses,
)


def test_discovery_across_roots_and_duplicate_error(tmp_path: Path) -> None:
    roots = [tmp_path / "m4", tmp_path / "cluster_b"]
    for root in roots:
        root.mkdir()
    (roots[0] / "run-a").mkdir()
    (roots[1] / "run-b").mkdir()
    rows = [{"run_id": "run-a"}, {"run_id": "run-b"}, {"run_id": "run-c"}]

    found = discover_run_dirs(rows, roots)
    assert found["run-a"].platform_root == roots[0]
    assert found["run-b"].run_dir == roots[1] / "run-b"
    assert "run-c" not in found

    (roots[1] / "run-a").mkdir()
    with pytest.raises(RuntimeError, match="run-a.*multiple --runs-root"):
        discover_run_dirs(rows, roots)


def test_incomplete_handling_is_opt_in(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    root.mkdir()
    rows = [{"run_id": "not-yet-run"}]
    located = discover_run_dirs(rows, [root])

    status = _verify(rows, located, allow_incomplete=True)
    assert status["not-yet-run"][0] == "MISSING"
    with pytest.raises(RuntimeError, match="required run.*missing or incomplete"):
        _verify(rows, located, allow_incomplete=False)


def _arrays(gap: float) -> dict[str, dict[str, np.ndarray]]:
    zero = np.zeros(4)
    higher = np.full(4, gap)
    return {
        "Jev-1.13.0": {"aff": zero, "net": zero},
        "DeepSeek-V4.1-Flash": {"aff": higher, "net": higher},
    }


def test_h1_h2_wiring_uses_each_eligibility_rule() -> None:
    blocks = np.arange(4)
    counts = stats.draws(4)
    assignment = {"block_s": 10.0, "n_blocks": 4, "blocks": blocks, "counts": counts}
    common = {
        "assignments": {"primary": assignment, "sensitivity_10s": assignment},
        "arrays": _arrays(0.2),
    }
    points = {
        BASE_POINT: common | {
            "is_rq2": False, "eligible_d10": True, "eligible_original_5pp_d10": False,
        },
        "dp-eligible-primary-only": common | {
            "is_rq2": True, "eligible_d10": True, "eligible_original_5pp_d10": False,
        },
        "dp-ineligible": common | {
            "is_rq2": True, "eligible_d10": False, "eligible_original_5pp_d10": False,
        },
    }

    result = hypotheses(points)
    assert result["primary_d10"]["H1"]["tested"] is True
    assert result["sensitivity_original_5pp"]["H1"]["tested"] is False
    primary_h2 = result["primary_d10"]["H2"]["result"]
    sensitivity_h2 = result["sensitivity_original_5pp"]["H2"]["result"]
    assert primary_h2["n_eligible"] == 1
    assert {test["design_point"] for test in primary_h2["tests"]} == {"dp-eligible-primary-only"}
    assert sensitivity_h2["n_eligible"] == 0
    assert sensitivity_h2["tests"] == []


def test_an_decomposition_signs_are_n_minus_l_and_n_minus_a() -> None:
    blocks = np.arange(4)
    counts = stats.draws(4)
    result = an_decomposition(
        n=np.full(4, 0.6),
        l=np.full(4, 0.2),
        a=np.full(4, 0.5),
        blocks=blocks,
        counts=counts,
    )
    assert result["n_minus_l"]["point"] == pytest.approx(0.4)
    assert result["n_minus_a"]["point"] == pytest.approx(0.1)
