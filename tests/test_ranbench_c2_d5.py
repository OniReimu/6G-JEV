"""Deviation D-5: actuating intents, intended direction, and C-1/C-2 on the signed affected-class measure.

Expected values are computed by hand in the comments, not with the code under test."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ranbench.c2 import stats
from src.ranbench.c2.outcomes import actuation, policy_rows_check

CLUSTERS = {"A": [1, 2], "B": [3], "all_cells": [1, 2, 3]}
PRIO = {"levels": {"critical": 10, "high": 30, "normal": 50, "low": 70}, "default_policy": "normal"}


def _ev(j: int, e: float, scope: str, cls: str, lvl: str) -> dict:
    return {"j": j, "intent_id": f"i{j}", "t_issue_s": e - 0.005, "e_s": e, "correct": True,
            "true_install": [scope, cls, lvl], "install": [scope, cls, lvl], "in_schedule": True}


def test_actuation_rule_and_direction() -> None:
    events = [
        _ev(0, 1.0, "A", "video", "critical"),  # 50 -> 10: raises priority, +1
        _ev(1, 2.0, "A", "video", "low"),       # 10 -> 70: lowers priority, -1
        _ev(2, 3.0, "B", "xr", "default"),      # 50 -> 50 (default = normal): no-op, 0
        _ev(3, 3.5, "A", "be", "critical"),     # BE: no SLA, 0
        _ev(4, 4.0, "B", "iot", "critical"),    # 50 -> 10: +1
        _ev(5, 5.0, "B", "iot", "default"),     # 10 -> 50: a revert that LOWERS priority, -1
        _ev(7, 6.0, "A", "xr", "high"),         # same e_s, applied after j6 (j order): 30 -> 30, 0
        _ev(6, 6.0, "A", "xr", "high"),         # 50 -> 30: +1
        _ev(8, 7.0, "all_cells", "video", "low"),  # cells 1,2 at 70, cell 3 at 50: sum(prior-new) = -20, -1
    ]
    rows = actuation(events, CLUSTERS, PRIO)
    assert [r["direction"] for r in rows] == [1, -1, 0, 0, 1, -1, 0, 1, -1]
    assert [r["actuating"] for r in rows] == [True, True, False, False, True, True, False, True, True]
    assert rows[8]["prior"] == [50, 70]


def test_actuation_ignores_installs_outside_the_schedule() -> None:
    ev = _ev(0, 1.0, "A", "video", "critical")
    ev["in_schedule"] = False
    assert actuation([ev], CLUSTERS, PRIO)[0] == {
        "j": 0, "cls": "video", "scope": "A", "level": "critical", "new": 10, "prior": None,
        "actuating": False, "direction": 0}


def test_policy_rows_check() -> None:
    events = [_ev(0, 1.0, "A", "video", "critical"), _ev(1, 2.0, "B", "xr", "default")]
    enf = pd.DataFrame({"t": [1.0, 1.0, 1.5, 2.0], "cell": [1, 2, 1, 3], "cls": ["video"] * 3 + ["xr"],
                        "old": [50, 48, 10, 52], "new": [10, 10, 12, 50], "source": ["policy", "policy", "xapp",
                                                                                     "policy"]})
    assert policy_rows_check(enf, events, CLUSTERS, PRIO) == {"n_policy_rows": 3, "n_mismatch": 0}
    enf.loc[1, "new"] = 30
    assert policy_rows_check(enf, events, CLUSTERS, PRIO)["n_mismatch"] == 1


def _blocks(n_blocks: int, per_block: int) -> tuple[np.ndarray, np.ndarray]:
    return np.repeat(np.arange(n_blocks), per_block), stats.draws(n_blocks)


def test_d5_controls_signed_mean_over_actuating_intents() -> None:
    # Per block: [raise, lower, no-op SLA intent, BE intent]; 4 identical blocks.
    direction = np.tile([1.0, -1.0, 0.0, 0.0], 4)
    nu = np.tile([0.6, 0.3, 0.9, np.nan], 4)
    orc = np.tile([0.2, 0.5, 0.0, np.nan], 4)
    fixed = {0.1: np.tile([0.2, 0.6, 0.0, np.nan], 4), 1.0: np.tile([0.3, 0.5, 0.0, np.nan], 4),
             5.0: np.tile([0.5, 0.3, 0.9, np.nan], 4)}
    blocks, counts = _blocks(4, 4)
    r = stats.d5_controls(nu, orc, fixed, direction, blocks, counts)
    # C-1: raise 0.6-0.2 = 0.4; lower 0.5-0.3 = 0.2; mean 0.3 (the no-op's 0.9 and the BE intent are excluded).
    assert r["C-1"]["point"] == pytest.approx(0.3) and r["C-1"]["n"] == 8 and r["C-1"]["n_actuating"] == 8
    assert r["C-1"]["ci_low"] == pytest.approx(0.3) and r["C-1"]["pass"] is True
    # C-2: raise 0.5-0.2 = 0.3; lower -(0.3-0.6) = 0.3; signed means d=0.1: (0.2-0.6)/2 = -0.2,
    # d=1: (0.3-0.5)/2 = -0.1, d=5: (0.5-0.3)/2 = 0.1 -> monotone.
    assert r["C-2"]["point"] == pytest.approx(0.3) and r["C-2"]["monotone"] is True and r["C-2"]["pass"] is True
    assert r["C-2"]["means"] == pytest.approx({0.1: -0.2, 1.0: -0.1, 5.0: 0.1})
    # Unsigned (all SLA intents): (0.4 - 0.2 + 0.9) / 3 = 0.3667 -- the rule must not return this.
    assert stats.c1_policy_matters(nu, orc, blocks, counts)["point"] == pytest.approx(1.1 / 3)


def test_d5_sign_handling_wrong_direction_fails() -> None:
    # Raise intent: oracle WORSE (0.4 vs 0.2) -> 0.2-0.4 = -0.2. Lower intent: oracle BETTER (0.3 vs 0.5), which is
    # against its intended direction -> 0.3-0.5 = -0.2. Signed mean -0.2 (the unsigned mean would be 0).
    direction = np.tile([1.0, -1.0], 4)
    nu, orc = np.tile([0.2, 0.5], 4), np.tile([0.4, 0.3], 4)
    fixed = {d: np.tile([0.2, 0.3], 4) for d in (0.1, 1.0, 5.0)}
    blocks, counts = _blocks(4, 2)
    r = stats.d5_controls(nu, orc, fixed, direction, blocks, counts)
    assert r["C-1"]["point"] == pytest.approx(-0.2) and r["C-1"]["pass"] is False
    assert stats.c1_policy_matters(nu, orc, blocks, counts)["point"] == pytest.approx(0.0)


@pytest.mark.parametrize("nu, passes", [(0.5, True), (0.49, False), (0.51, True)])
def test_d5_threshold_boundary(nu: float, passes: bool) -> None:
    # Gap nu - 0.45: 0.05 exactly (0.04999999999999999 in floating point) passes; 0.04 fails; 0.06 passes.
    direction = np.ones(4)
    fixed = {d: np.full(4, 0.3) for d in (0.1, 1.0, 5.0)}
    r = stats.d5_controls(np.full(4, nu), np.full(4, 0.45), fixed, direction, *_blocks(4, 1))
    assert r["C-1"]["point"] == pytest.approx(nu - 0.45)
    assert r["C-1"]["pass"] is passes


def test_d5_no_actuating_intents_is_undefined_and_fails() -> None:
    fixed = {d: np.full(4, 0.3) for d in (0.1, 1.0, 5.0)}
    r = stats.d5_controls(np.full(4, 0.9), np.full(4, 0.1), fixed, np.zeros(4), *_blocks(4, 1))
    assert r["C-1"]["point"] is None and r["C-1"]["n_actuating"] == 0 and r["C-1"]["pass"] is False
    assert r["C-2"]["pass"] is False
