"""Stale-policy exposure (descriptive C2 KPI): hand-computable fixtures."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ranbench.c2 import schedule as sch
from src.ranbench.c2.loader import load_run
from src.ranbench.c2.outcomes import actuation, stale_policy_exposure

SIM_S = 30.0
CONFIG = {
    "clusters": {"stadium": [1], "north_cluster": [2]},
    "priorities": {"levels": {"critical": 10, "high": 30, "normal": 50, "low": 70}, "default_policy": "normal"},
    "traffic": {"classes": {"video": {"target_type": "throughput_floor_kbps", "target_value": 1000.0},
                            "xr": {"target_type": "delay_p95_ms", "target_value": 20.0}}},
    "simulation": {"sim_time_s": SIM_S},
}


def write_run(d: Path, schedule_rows: list[tuple]) -> Path:
    """Stadium (cell 1) serves video UEs 1 and 2 and xr UE 4 throughout; video UE 3 is in the north cell for the
    slot ending at 10.1 and in the stadium from the slot ending at 10.2 on."""
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(CONFIG))
    rows = []
    for k in range(1, int(round(SIM_S / 0.1)) + 1):
        t = round(k * 0.1, 3)
        for imsi, cls, cell in ((1, "video", 1), (2, "video", 1), (3, "video", 2 if t <= 10.1 else 1),
                                (4, "xr", 1)):
            rows.append(f"{t:.3f},{imsi},{cell},{cls},1250,10,5.000,10.000")
    (d / "ue_slots.csv").write_text(
        "t_slot,imsi,serving_cell,class,rx_bytes,n_rx_pkts,mean_delay_ms,p95_delay_ms\n" + "\n".join(rows) + "\n")
    sch.write_schedule(schedule_rows, d / "schedule.csv")
    return d


def event(j: int, t: float, e: float | None, level: str = "high", install: str | None = None) -> dict:
    true_install = ["stadium", "video", level]
    inst = true_install if install is None else ["stadium", "video", install]
    return {"j": j, "t_issue_s": t, "e_s": e, "true_install": true_install,
            "install": inst if e is not None else None, "in_schedule": e is not None and e <= SIM_S}


def test_two_ues_one_intent_known_latency(tmp_path):
    # issued at 10.05, in the slot ending at 10.1: stadium video UEs 1 and 2 (UE 3 arrives in the next slot)
    tr = load_run(write_run(tmp_path / "L", [(10.55, "stadium", "video", "high")]))
    out = stale_policy_exposure(tr, [event(0, 10.05, 10.55)], {0})
    assert out["mean_ue_s"] == pytest.approx(2 * 0.5)
    assert out["median_delay_s"] == pytest.approx(0.5) and out["n_actuating"] == 1


def test_cap_at_next_intent_on_same_class_scope_and_at_run_end(tmp_path):
    # j=0 installs at 13.0, after j=1 on the same (stadium, video) is issued at 12.05: capped there (2 UEs x 2.0 s).
    # j=1 lands after run end (not in the schedule): capped at 30.0, with UEs 1, 2, 3 in the stadium (3 x 17.95 s).
    tr = load_run(write_run(tmp_path / "L", [(13.0, "stadium", "video", "high")]))
    out = stale_policy_exposure(tr, [event(0, 10.05, 13.0), event(1, 12.05, 40.0, level="low")], {0, 1})
    assert out["mean_ue_s"] == pytest.approx((2 * 2.0 + 3 * 17.95) / 2)
    assert out["median_delay_s"] == pytest.approx((2.0 + 17.95) / 2)
    # a wrong policy on the pair is not j's policy: capped at run end
    tr = load_run(write_run(tmp_path / "N", [(10.55, "stadium", "video", "low")]))
    out = stale_policy_exposure(tr, [event(0, 10.05, 10.55, install="low")], {0})
    assert out["mean_ue_s"] == pytest.approx(2 * (SIM_S - 10.05))


def test_non_actuating_intent_is_excluded(tmp_path):
    # oracle arm: j=1 repeats j=0's level on the same pair, so D-5 marks only j=0 as actuating
    oracle = [event(0, 10.05, 10.055), event(1, 20.05, 20.055)]
    act = actuation(oracle, CONFIG["clusters"], CONFIG["priorities"])
    actuating = {row["j"] for row in act if row["actuating"]}
    assert actuating == {0}
    tr = load_run(write_run(tmp_path / "L", [(10.55, "stadium", "video", "high"),
                                             (25.0, "stadium", "video", "high")]))
    out = stale_policy_exposure(tr, [event(0, 10.05, 10.55), event(1, 20.05, 25.0)], actuating)
    assert out["n_actuating"] == 1
    assert out["mean_ue_s"] == pytest.approx(2 * 0.5) and out["median_delay_s"] == pytest.approx(0.5)
