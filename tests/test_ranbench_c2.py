"""EXP-2026-003 C2 Python side: streams, enforcement schedules, trace outcomes, controls, bootstrap and H1/H2.

Synthetic world with known answers: two cells (stadium = cell 1, north_cluster = cell 2). Every intent is a demand
event for its (scope, class): from its issue time the class's video UEs in scope miss the 5 Mbps floor until a
high/critical level installed after that issue time is in effect (responsive=True). With responsive=False the radio
ignores the schedule, so latency cannot change any SLA outcome.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.ranbench.c2 import schedule as sch
from src.ranbench.c2 import stats
from src.ranbench.c2.loader import load_run, load_table
from src.ranbench.c2.matrix import INTERPRETERS, generate_matrix, totals, wall_hours, write_matrix
from src.ranbench.c2.outcomes import SlotTable, intent_outcomes, network_series, radio_kpis
from src.ranbench.c2.streams import (
    N_EVENTS_MIN, T0_S, TAIL_S, load_stream, make_stream, n_events, run_length_s, stream_from_trace, write_stream,
)

CMAP = sch.class_map()
DE2 = sch.DELTA_E2_S
CONFIG = {
    "clusters": {"_rule": "test", "stadium": [1], "north_cluster": [2], "all_cells": [1, 2]},
    "traffic": {"classes": {
        "video": {"target_type": "throughput_floor_kbps", "target_value": 5000.0},
        "xr": {"target_type": "delay_p95_ms", "target_value": 20.0},
        "iot": {"target_type": "throughput_floor_kbps", "target_value": 80.0},
        "be": {"target_type": "throughput_floor_kbps", "target_value": 10000.0},
    }},
    "simulation": {"sim_time_s": 0.0},
}
# imsi, class, cell (static)
UES = [(1, "video", 1), (2, "video", 1), (3, "xr", 1), (4, "be", 1), (5, "video", 2), (6, "iot", 2)]


def pool_record(i: int, scope: str = "stadium", cls: str = "emergency_video", action: str = "prioritise",
                priority: str = "high") -> dict:
    act = {"action": action, "class": cls, "scope": scope, "priority": priority}
    return {"intent_id": f"i{i:03d}", "actuated": act, "c2_class": CMAP[cls], "truth": act}


def stream_of(times: list[float], pools: list[dict], sim_time: float, block_s: float = 10.0,
              length: float | None = None) -> dict:
    length = length if length is not None else max(times) - T0_S + 1.0
    events = [{"j": j, "intent_id": p["intent_id"], "t_issue_s": t, "actuated": p["actuated"],
               "c2_class": p["c2_class"]} for j, (t, p) in enumerate(zip(times, pools))]
    return {"meta": {"key": "test", "sim_time_s": sim_time, "t0_s": T0_S, "block_s": block_s,
                     "run_length_s": length, "n_blocks": int(length // block_s)}, "events": events}


def write_run(d: Path, stream: dict, rows: list[tuple], responsive: bool = True, mobile_imsi: int | None = None,
              handovers: str = "t_start,t_end_ok,imsi,source_cell,target_cell,handover_total_time_ms,outcome\n") -> Path:
    """Traces of one arm run; rows = the arm's schedule (t_s, scope, class, level)."""
    d.mkdir(parents=True, exist_ok=True)
    sim = stream["meta"]["sim_time_s"]
    cfg = json.loads(json.dumps(CONFIG))
    cfg["simulation"]["sim_time_s"] = sim
    (d / "config.json").write_text(json.dumps(cfg))
    cl = cfg["clusters"]
    needs = [(e["t_issue_s"], cl[e["actuated"]["scope"]], e["c2_class"]) for e in stream["events"]]
    highs = [(t, cl[s], c) for t, s, c, lev in rows if lev in ("critical", "high")]
    n_slots = int(round(sim / 0.1))
    ue_rows, pos_rows, tx_rows, prb_rows = [], [], [], []
    for k in range(1, n_slots + 1):
        ts = round(k * 0.1, 3)
        for imsi, cls, cell0 in UES:
            cell = cell0
            if imsi == mobile_imsi:
                cell = 1 if int(ts // 7) % 2 == 0 else 2
            kbps = offered = {"video": 6000.0, "xr": 2000.0, "iot": 100.0, "be": 20000.0}[cls]
            if responsive and cls == "video":
                need = max([t for t, cells, c in needs if c == cls and cell in cells and t < ts], default=None)
                if need is not None:
                    ok = any(need <= t <= ts and cell in cells and c == cls for t, cells, c in highs)
                    kbps = 6000.0 if ok else 3000.0
            nbytes = int(kbps * 1000 * 0.1 / 8)
            npk = 10
            ue_rows.append(f"{ts:.3f},{imsi},{cell},{cls},{nbytes},{npk},5.000,10.000")
            pos_rows.append(f"{ts:.3f},{imsi},0.000,0.000,{cell}")
            tx_rows.append(f"{ts:.3f},{imsi},{int(offered * 1000 * 0.1 / 8)},{npk}")
        for cell in (1, 2):
            prb_rows.append(f"{ts:.3f},{cell},0.5000")
    (d / "ue_slots.csv").write_text(
        "t_slot,imsi,serving_cell,class,rx_bytes,n_rx_pkts,mean_delay_ms,p95_delay_ms\n" + "\n".join(ue_rows) + "\n")
    (d / "positions.csv").write_text("t,imsi,x,y,serving_cell\n" + "\n".join(pos_rows) + "\n")
    (d / "tx_trace.csv").write_text("t_slot,imsi,tx_bytes,n_tx_pkts\n" + "\n".join(tx_rows) + "\n")
    (d / "cell_prb.csv").write_text("t_slot,cell,prb_share\n" + "\n".join(prb_rows) + "\n")
    (d / "handover.csv").write_text(handovers)
    (d / "rlf.csv").write_text("t,imsi,cell,cause\n1.0,1,1,T310\n5.0,2,1,T310\n")
    (d / "enforcement.csv").write_text("t,cell,class,old,new,source,extra_col\n")
    sch.write_schedule(rows, d / "schedule.csv")
    return d


def records_for(stream: dict, model: str, latency: float, policy=None, valid: bool = True) -> dict:
    return {e["intent_id"]: {"case_id": e["intent_id"], "model": model, "valid": valid, "latency_s": latency,
                             "labels": dict(policy or e["actuated"])} for e in stream["events"]}


def run_arm(tmp: Path, name: str, stream: dict, arm: str, responsive: bool = True, **kw) -> tuple[dict, list[dict]]:
    a = sch.build_arm(stream, arm, cmap=CMAP, **kw)
    tr = load_run(write_run(tmp / name, stream, a["schedule"], responsive))
    return a, intent_outcomes(tr, a["events"])


# ------------------------------------------------------------------ streams

def test_stream_counts_times_and_determinism(tmp_path):
    pool = [pool_record(i) for i in range(150)]
    for rate, block, n in [(0.1, 10, 100), (0.3, 10, 100), (1.0, 10, 100), (1.0, 20, 200), (0.1, 150, 150)]:
        s = make_stream(pool, rate, block, key=f"r{rate}")
        assert s["meta"]["n_events"] == n == len(s["events"]) == n_events(rate, block)
        t = np.array([e["t_issue_s"] for e in s["events"]])
        assert (np.diff(t) >= 0).all() and t[0] >= T0_S and t[-1] < T0_S + run_length_s(rate, block)
        assert s["meta"]["n_blocks"] >= 10
        assert s["meta"]["sim_time_s"] == pytest.approx(T0_S + run_length_s(rate, block) + TAIL_S)
    s1, s2 = make_stream(pool, 0.3, 10, "a"), make_stream(pool, 0.3, 10, "a")
    assert s1 == s2 and s1 != make_stream(pool, 0.3, 10, "b")
    assert len({e["intent_id"] for e in s1["events"]}) == N_EVENTS_MIN  # no repeat within the first pass
    big = make_stream(pool, 1.0, 20, "c")
    assert len({e["intent_id"] for e in big["events"][:150]}) == 150  # full pass before any repeat
    # Poisson: exponential gaps have CV ~ 1
    gaps = np.diff([e["t_issue_s"] for e in make_stream(pool, 1.0, 100, "d")["events"]])
    assert 0.8 < gaps.std() / gaps.mean() < 1.2


def test_stream_sha_written_verified_and_frozen(tmp_path):
    s = make_stream([pool_record(i) for i in range(5)], 0.3, 10, "k")
    p = tmp_path / "stream.json"
    sha = write_stream(s, p)
    assert sha == hashlib.sha256(p.read_bytes()).hexdigest()
    assert load_stream(p) == s
    assert write_stream(s, p) == sha  # identical rewrite is fine
    with pytest.raises(ValueError, match="frozen"):
        write_stream(make_stream([pool_record(i) for i in range(5)], 0.3, 10, "other"), p)
    p.write_bytes(p.read_bytes().replace(b'"k"', b'"x"'))
    with pytest.raises(ValueError, match="sha256"):
        load_stream(p)


def test_stream_from_trace_keeps_arrival_spacing():
    pool = [pool_record(i) for i in range(5)]
    rows = [{"intent_id": f"i{i % 5:03d}", "arrival_s": 100 + 0.5 * i} for i in range(120)]
    s = stream_from_trace(rows, pool, 10.0, "rq3")
    t = [e["t_issue_s"] for e in s["events"]]
    assert len(t) == 100 and t[0] == T0_S and np.allclose(np.diff(t), 0.5)


# ------------------------------------------------------------------ production matrix

def test_production_matrix_is_complete_unique_and_single_seed(tmp_path):
    rows = generate_matrix(tmp_path / "inputs", block_s=10.0)
    assert len(rows) == 316
    assert len({row.run_id for row in rows}) == len(rows)
    assert {row.rng_run for row in rows} == {1}
    assert {row.priority_tier for row in rows} == {1, 2, 3}
    assert all(row.ues == 21 * row.ues_per_cell for row in rows)
    assert sum(row.priority_tier == 1 for row in rows) == 120
    assert sum(row.priority_tier == 2 for row in rows) == 189
    assert sum(row.priority_tier == 3 for row in rows) == 7

    rq2_l = [row for row in rows if row.rqs == "RQ2" and row.arm == "L" and row.priority_tier == 1]
    assert len(rq2_l) == 3 * 3 * len(INTERPRETERS)
    assert all(row.availability == "waiting_for_c1_ledger" for row in rq2_l)
    controls = [row for row in rows if row.rqs == "RQ2" and row.interpreter == "control"]
    assert len(controls) == 3 * 3 * 5
    assert all(row.availability == "ready_after_pilot" for row in controls)
    rq1_e = [row for row in rows if row.rqs == "RQ1" and row.mode == "E" and row.arm == "L"]
    assert len(rq1_e) == len(INTERPRETERS) and all(row.priority_tier == 1 for row in rq1_e)
    rq3 = [row for row in rows if row.rqs == "RQ3"]
    assert len(rq3) == len(INTERPRETERS) * 5
    assert sum(row.ues_per_cell == 20 for row in rq3) == len(INTERPRETERS)
    assert all(row.availability == "waiting_for_load_trace" for row in rq3)
    rq5 = [row for row in rows if row.priority_tier == 3]
    assert len(rq5) == len(INTERPRETERS)
    assert {row.arm for row in rq5} == {"b-requery-1s"}
    assert {row.interpreter for row in rq5} == set(INTERPRETERS)
    assert {row.reused_arm_c_run_id for row in rq5} == {"rq1_e_r0p3_s30_oracle"}
    assert all(row.reused_arm_a_run_id == f"rq2_e_r0p3_s30_n_{row.run_id.split('b_requery_1s_', 1)[1]}"
               for row in rq5)
    assert all(row.estimated_wall_hours_mac_m4_max > 0 for row in rq5)
    assert totals(rows)["rows_by_tier"] == {"1": 120, "2": 189, "3": 7}
    # The base point (0.3/s, 30 km/h) is off the RQ2 grid and carries its own five controls.
    base = [row for row in rows if row.design_point == "dp_r0p3_s30_u5"]
    assert sorted(row.arm for row in base if row.interpreter == "control") == sorted(
        ["oracle", "fixed-0.1", "fixed-1", "fixed-5", "no-update"])
    assert all(row.priority_tier == 1 for row in base if row.interpreter == "control")
    assert {row.design_point for row in rq5} == {"dp_r0p3_s30_u5"}
    # One design point = one stream, one config, one UE count.
    for key in {row.design_point for row in rows if row.rqs != "RQ5"}:
        members = [row for row in rows if row.design_point == key and row.rqs != "RQ5"]
        assert len({(row.stream_file, row.config_file, row.ues) for row in members}) == 1, key


def test_rq2_an_arms_cover_grid_without_per_mode_controls(tmp_path):
    rows = generate_matrix(tmp_path / "inputs", block_s=10.0)
    rq2_an = [row for row in rows if row.rqs == "RQ2" and row.arm in ("A", "N")]
    grid = [row for row in rq2_an if row.speed_kmh in (3.0, 60.0, 120.0)]
    assert len(grid) == 2 * 3 * 3 * len(INTERPRETERS)
    assert {(row.rate_per_s, row.speed_kmh, row.interpreter, row.arm) for row in grid} == {
        (rate, speed, interpreter, arm)
        for rate in (0.1, 0.3, 1.0)
        for speed in (3.0, 60.0, 120.0)
        for interpreter in INTERPRETERS
        for arm in ("A", "N")
    }
    assert all(row.mode == "E" and row.priority_tier == 2
               and row.availability == "waiting_for_c1_ledger" for row in grid)
    assert not [row for row in rows if row.interpreter == "control" and row.mode != "E"]


def test_production_matrix_cost_model_and_no_overwrite(tmp_path):
    assert wall_hours(60.0, 105) == pytest.approx(0.89 * 105 * 60 / 3600)
    assert wall_hours(60.0, 420) / wall_hours(60.0, 105) == pytest.approx(8.0)
    rows = generate_matrix(tmp_path / "inputs")
    out = tmp_path / "matrix.csv"
    write_matrix(rows, out)
    with pytest.raises(FileExistsError, match="overwrite"):
        write_matrix(rows, out)


# ------------------------------------------------------------------ schedules

def test_enforcement_times_modes():
    t, lat = np.array([1.0, 2.0]), np.array([0.2, np.nan])
    assert sch.enforcement_times("X", t, lat)[0] == pytest.approx(1.2 + DE2)
    e = sch.enforcement_times("E", t, lat, d_a1=sch.delta_a1(0.010, allow_override=True))
    assert e[0] == pytest.approx(1.2 + 0.015 + DE2) and np.isinf(e[1])
    with pytest.raises(ValueError, match="delta_A1"):
        sch.enforcement_times("E", t, lat)


def test_periodic_four_slots_and_no_overlap():
    # five intents in cycle (0, 10], 3 s each: four finish at 13, the fifth takes the first free slot -> 16
    t = np.array([1, 2, 3, 4, 5, 12.0])
    lat = np.full(6, 3.0)
    assert list(sch.periodic_ready(t, lat, 10.0)) == [13, 13, 13, 13, 16, 23]
    # eight intents of 12 s: batch ends at 34 > next grid start 20, so the next cycle starts at 34 (no overlap)
    t = np.array([1, 2, 3, 4, 5, 6, 7, 8, 25.0, 36.0])
    lat = np.array([12] * 8 + [1.0, 1.0])
    r = sch.periodic_ready(t, lat, 10.0)
    assert list(r[:8]) == [22] * 4 + [34] * 4
    assert r[8] == 35  # issued at 25, waits for the delayed start at 34
    assert r[9] == 41  # next grid start after 34 is 40
    # issued exactly at a cycle start joins that cycle; an empty cycle is skipped
    assert list(sch.periodic_ready(np.array([10.0, 31.0]), np.array([1.0, 1.0]), 10.0)) == [11, 41]
    e = sch.enforcement_times("P-10", np.array([1.0]), np.array([0.5]), d_a1=0.01)
    assert e[0] == pytest.approx(10.5 + 0.01 + DE2)


def test_install_and_correctness_rules():
    assert sch.install_of({"action": "prioritise", "class": "xr_gaming", "scope": "stadium", "priority": "critical"},
                          CMAP) == ("stadium", "xr", "critical")
    assert sch.install_of({"action": "revert_default", "class": "iot_metering", "scope": "north_cluster",
                           "priority": "unspecified"}, CMAP) == ("north_cluster", "iot", "default")
    for bad in [{"action": "guarantee_share", "class": "best_effort", "scope": "stadium", "priority": "high"},
                {"action": "prioritise", "class": "best_effort", "scope": "most_loaded_cells", "priority": "high"},
                {"action": "prioritise", "class": "best_effort", "scope": "stadium", "priority": "unspecified"},
                None]:
        assert sch.install_of(bad, CMAP) is None
    truth = pool_record(0)["actuated"]
    assert sch.is_correct(dict(truth, prb_share="10"), truth)
    assert not sch.is_correct(dict(truth, action="deprioritise"), truth)  # same install, wrong vector


def test_build_arms(tmp_path):
    pools = [pool_record(0), pool_record(1, scope="north_cluster", cls="best_effort", action="revert_default",
                                        priority="unspecified"), pool_record(2)]
    s = stream_of([10.0, 20.0, 29.9], pools, sim_time=30.0)
    wrong = {"action": "prioritise", "class": "iot_metering", "scope": "north_cluster", "priority": "high"}
    recs = records_for(s, "m", 0.5)
    recs["i000"]["labels"] = wrong
    recs["i001"]["valid"] = False
    L = sch.build_arm(s, "L", "X", recs, cmap=CMAP)
    assert L["schedule"] == [(10.505, "stadium", "video", "high"), (20.505, "north_cluster", "be", "default")]
    assert all(e["correct"] for e in L["events"]) and not L["events"][2]["in_schedule"]  # 30.405 > sim end
    A = sch.build_arm(s, "A", "X", recs, cmap=CMAP)
    assert A["schedule"][0] == (10.005, "north_cluster", "iot", "high")  # wrong policy is installed
    assert [e["correct"] for e in A["events"]] == [False, False, True]
    assert A["events"][1]["install"] is None  # invalid output installs nothing
    N = sch.build_arm(s, "N", "E", recs, d_a1=0.01, cmap=CMAP)
    assert N["events"][0]["e_s"] == pytest.approx(10.5 + 0.015)
    assert sch.build_arm(s, "oracle", cmap=CMAP)["schedule"][0][0] == pytest.approx(10.005)
    assert sch.build_arm(s, "no_update", cmap=CMAP)["schedule"] == []
    F = sch.build_arm(s, "fixed", "E", fixed_d=5.0, d_a1=0.01, cmap=CMAP)
    assert F["events"][0]["e_s"] == pytest.approx(15.015)
    with pytest.raises(ValueError, match="no record"):
        sch.build_arm(s, "L", "X", {}, cmap=CMAP)
    sch.write_arm(L, tmp_path / "L")
    lines = (tmp_path / "L" / "schedule.csv").read_text().splitlines()
    assert lines == ["t_s,scope,class,priority", "10.505000,stadium,video,high", "20.505000,north_cluster,be,default"]


# ------------------------------------------------------------------ outcomes

def _one_intent_stream():
    return stream_of([10.05], [pool_record(0)], sim_time=25.0)


def test_latency_changes_sla_outcome(tmp_path):
    s = _one_intent_stream()
    _, fast = run_arm(tmp_path, "fast", s, "L", mode="X", records=records_for(s, "fast", 0.3))
    _, slow = run_arm(tmp_path, "slow", s, "L", mode="X", records=records_for(s, "slow", 1.5))
    # window (10.05, 12.05] = slots 10.1..12.0; stadium video misses in slots ending before e
    assert fast[0]["aff_W2"] == pytest.approx(3 / 20)   # e = 10.355: slots 10.1, 10.2, 10.3
    assert slow[0]["aff_W2"] == pytest.approx(15 / 20)  # e = 11.555: slots 10.1 .. 11.5
    assert fast[0]["net_W2"] == pytest.approx(2 * 3 / (6 * 20))
    assert slow[0]["net_W2"] == pytest.approx(2 * 15 / (6 * 20))
    assert slow[0]["aff_W1"] == 1.0 and slow[0]["aff_W10"] == pytest.approx(15 / 100)
    assert fast[0]["E_s"] == pytest.approx(0.305) and slow[0]["E_s"] == pytest.approx(1.505)
    assert fast[0]["S_ue_s"] == pytest.approx(2 * 0.305) and slow[0]["S_ue_s"] == pytest.approx(2 * 1.505)
    assert fast[0]["aff_life"] == 0.0 and fast[0]["collateral_ue_s"] == 0.0


def test_latency_does_not_change_sla_when_radio_ignores_policy(tmp_path):
    s = _one_intent_stream()
    _, fast = run_arm(tmp_path, "fast", s, "L", responsive=False, mode="X", records=records_for(s, "f", 0.3))
    _, slow = run_arm(tmp_path, "slow", s, "L", responsive=False, mode="X", records=records_for(s, "s", 1.5))
    for w in (1, 2, 5, 10):
        assert fast[0][f"aff_W{w}"] == slow[0][f"aff_W{w}"] == 0.0
        assert fast[0][f"net_W{w}"] == slow[0][f"net_W{w}"] == 0.0
    assert slow[0]["S_ue_s"] > fast[0]["S_ue_s"]  # exposure is accounting, not radio: it still grows with latency


def test_wrong_policy_never_enforced_and_collateral(tmp_path):
    s = _one_intent_stream()
    wrong = {"action": "prioritise", "class": "video_streaming", "scope": "north_cluster", "priority": "critical"}
    a, rows = run_arm(tmp_path, "A", s, "A", records=records_for(s, "m", 0.3, policy=wrong))
    r = rows[0]
    assert not r["correct"] and np.isinf(r["E_s"])
    assert r["aff_W2"] == 1.0  # truth never enforced; the video UEs in the stadium miss throughout
    assert r["S_ue_s"] == pytest.approx(2 * (25.0 - 10.05))  # until run end (no later intent on the same pair)
    assert r["collateral_ue_s"] == pytest.approx(1 * (25.0 - 10.055))  # one video UE in north, from e to run end
    assert r["aff_life"] == 1.0


def test_exposure_stops_at_next_intent_on_same_pair(tmp_path):
    s = stream_of([10.05, 12.05], [pool_record(0), pool_record(1)], sim_time=25.0)
    _, rows = run_arm(tmp_path, "A", s, "A", records=records_for(s, "m", 0.3, valid=False))
    assert rows[0]["S_ue_s"] == pytest.approx(2 * 2.0) and rows[1]["S_ue_s"] == pytest.approx(2 * (25 - 12.05))


def test_be_intent_has_no_affected_class_outcome(tmp_path):
    s = stream_of([10.05], [pool_record(0, cls="best_effort")], sim_time=25.0)
    _, rows = run_arm(tmp_path, "o", s, "oracle")
    assert rows[0]["aff_W2"] is None and rows[0]["net_W2"] == 0.0


def test_radio_kpis_known_values(tmp_path):
    s = stream_of([10.05], [pool_record(0)], sim_time=20.0)
    ho = ("t_start,t_end_ok,imsi,source_cell,target_cell,handover_total_time_ms,outcome\n"
          "3.0,3.05,1,1,2,50.0,ok\n4.0,4.1,2,1,2,100.0,ok\n5.0,,3,1,2,,end_error\n")
    tr = load_run(write_run(tmp_path / "k", s, [], responsive=False, handovers=ho))
    k = radio_kpis(tr, 2.0, 20.0)
    assert k["thr_mean_mbps.video"] == pytest.approx(6.0) and k["thr_mean_mbps.iot"] == pytest.approx(0.1)
    assert k["outage_share"] == 0.0 and k["prb_util_mean"] == pytest.approx(0.5)
    assert k["ho_count"] == 2 and k["ho_fail_count"] == 1
    assert k["ho_rate_per_ue_s"] == pytest.approx(2 / (6 * 18.0))
    assert k["ho_total_time_p50_ms"] == pytest.approx(75.0)
    assert k["rlf_count"] == 1 and k["rlf_count_total"] == 2
    assert k["delay_p50_ms"] == 5.0 and k["jitter_iqr_ms"] == 0.0


def test_radio_kpis_p5_is_across_per_ue_means(tmp_path):
    s = stream_of([10.05], [pool_record(0)], sim_time=20.0)
    d = write_run(tmp_path / "p5", s, [], responsive=False)
    # UE 1 is idle in 9 of 10 slots (a 10 Mb/s burst every 10th slot: mean 1 Mb/s); the others are steady.
    steady = {2: 4.0, 3: 2.0, 4: 8.0, 5: 6.0, 6: 0.5}
    rows = []
    for k in range(1, 201):
        for imsi, cls, cell in UES:
            mbps = (10.0 if k % 10 == 0 else 0.0) if imsi == 1 else steady[imsi]
            rows.append(f"{k * 0.1:.3f},{imsi},{cell},{cls},{int(mbps * 1e6 * 0.1 / 8)},10,5.000,10.000")
    (d / "ue_slots.csv").write_text(
        "t_slot,imsi,serving_cell,class,rx_bytes,n_rx_pkts,mean_delay_ms,p95_delay_ms\n" + "\n".join(rows) + "\n")
    k = radio_kpis(load_run(d), 0.0, 20.0)
    assert k["thr_p5_mbps"] > 0.0  # more than 5% of (UE, slot) pairs are idle, yet every UE's mean is positive
    assert k["thr_p5_mbps"] == pytest.approx(np.percentile([1.0, 4.0, 2.0, 8.0, 6.0, 0.5], 5))
    assert k["thr_p5_mbps.video"] == pytest.approx(np.percentile([1.0, 4.0, 6.0], 5))
    assert k["thr_mean_mbps"] == pytest.approx(np.mean([1.0, 4.0, 2.0, 8.0, 6.0, 0.5]))  # slot mean, unchanged


# ------------------------------------------------------------------ loader

def test_loader_aliases_extra_columns_torn_tail(tmp_path):
    d = tmp_path / "r"
    d.mkdir()
    (d / "handover.csv").write_text("t_start,t_end,imsi,source_cell,target_cell,handover_total_time_ms\n1,1.05,4,1,2,50\n")
    h = load_table(d, "handover")
    assert list(h.columns) == ["t_start", "t_end", "imsi", "source", "target", "total_ms"] and h["t_end"][0] == 1.05
    (d / "rlf.csv").write_text("")
    assert len(load_table(d, "rlf")) == 0
    (d / "cell_prb.csv").write_text("t_slot,cell,prb_share\n0.1,1,0.5\n0.2,1,0.")
    with pytest.raises(ValueError, match="newline"):
        load_table(d, "cell_prb")
    assert len(load_table(d, "cell_prb", tolerate_torn_tail=True)) == 1
    (d / "cell_prb.csv").write_text("t,cell,prb\n")
    with pytest.raises(ValueError, match="none of"):
        load_table(d, "cell_prb")


# ------------------------------------------------------------------ controls

def test_c3_exposure_linear_in_d(tmp_path):
    # spread over all 12 (scope, class) pairs so S_j is rarely capped by the next intent on the same pair
    combos = [(sc, c) for sc in ("stadium", "north_cluster", "all_cells")
              for c in ("emergency_video", "xr_gaming", "iot_metering", "best_effort")]
    pool = [pool_record(i, scope=combos[i % 12][0], cls=combos[i % 12][1]) for i in range(150)]
    s = make_stream(pool, 0.1, 10.0, key="c3")
    tr = load_run(write_run(tmp_path / "static", s, [], responsive=False))
    d_a1, ds = 0.01, (0.1, 1.0, 5.0)

    # Closed form from the fixture, independent of SlotTable: static UEs, so the in-scope count of an intent is
    # the number of UES rows of its class in its scope's cells, and S_j(d) = n_j x (min(e_j(d), next issue on the
    # same (class, scope), run end) - t_j) with e_j(d) = t_j + d + delta_A1 + delta_E2 (mode E).
    cells = CONFIG["clusters"]
    ev = s["events"]
    n = [sum(1 for _, c, cell in UES if c == e["c2_class"] and cell in cells[e["actuated"]["scope"]]) for e in ev]
    t = [e["t_issue_s"] for e in ev]
    nxt = [min([t[k] for k in range(j + 1, len(ev)) if (ev[k]["c2_class"], ev[k]["actuated"]["scope"])
                == (ev[j]["c2_class"], ev[j]["actuated"]["scope"])], default=np.inf) for j in range(len(ev))]
    t_end = s["meta"]["sim_time_s"]
    s_hand = {d: [n[j] * (min(t[j] + d + d_a1 + DE2, nxt[j], t_end) - t[j]) for j in range(len(ev))] for d in ds}
    n_hand = float(np.mean(n))
    slope_hand = np.polyfit(ds, [np.mean(s_hand[d]) for d in ds], 1)[0]
    assert 0 < slope_hand < n_hand  # capping at the next same-pair intent only lowers the slope

    for d in ds:  # per-intent S_j from the production path equals the closed form
        rows = intent_outcomes(tr, sch.build_arm(s, "fixed", "E", fixed_d=d, d_a1=d_a1, cmap=CMAP)["events"])
        assert [r["S_ue_s"] for r in rows] == pytest.approx(s_hand[d], abs=1e-6)
    res = stats.c3_from_traces(tr, s, d_values=ds, d_a1=d_a1, cmap=CMAP)
    assert res["mean_in_scope_ues"] == pytest.approx(n_hand)
    assert res["slope"] == pytest.approx(slope_hand)
    assert res["pass"] == (abs(slope_hand - n_hand) / n_hand <= 0.10) and res["pass"], res
    # endpoints are chosen by value, not position
    assert stats.c3_from_traces(tr, s, d_values=(5.0, 0.1, 1.0), d_a1=d_a1, cmap=CMAP) == res

    tr = load_run(write_run(tmp_path / "mobile", s, [], responsive=False, mobile_imsi=5))
    assert stats.c3_from_traces(tr, s, d_values=ds, d_a1=d_a1, cmap=CMAP)["pass"]
    bad = stats.c3_accounting([0.1, 1.0, 5.0], [0.2, 2.0, 10.0], 3.0)  # slope 2 vs 3 UEs
    assert not bad["pass"]


def test_c4_pairing_bytewise(tmp_path):
    s = _one_intent_stream()
    a = write_run(tmp_path / "oracle", s, sch.build_arm(s, "oracle", cmap=CMAP)["schedule"])
    b = write_run(tmp_path / "none", s, [])
    res = stats.c4_pairing(a, b)
    assert res["files"] == {"positions.csv": True, "tx_trace.csv": True}  # response only changes rx
    # the fixture's mobile UE changes serving_cell only (coordinates stay 0, 0): positions pair (D-6 C-4 columns);
    # a coordinate difference fails, see test_c4_pairing_ignores_serving_cell_but_not_positions
    assert stats.c4_pairing(a, write_run(tmp_path / "moved", s, [], mobile_imsi=5))["pass"]


def _base_point(tmp_path: Path, responsive: bool = True):
    pool = [pool_record(i, scope=["stadium", "north_cluster"][i % 2]) for i in range(150)]
    s = make_stream(pool, 0.3, 10.0, key="base")
    t = np.array([e["t_issue_s"] for e in s["events"]])
    blocks = stats.block_ids(t, s["meta"]["t0_s"], 10.0, s["meta"]["n_blocks"])
    counts = stats.draws(s["meta"]["n_blocks"])

    def arm(name, **kw):
        return run_arm(tmp_path, name, s, responsive=responsive, **kw)[1]

    return s, blocks, counts, arm


def test_controls_c1_c2_pass_when_radio_responds(tmp_path):
    s, blocks, counts, arm = _base_point(tmp_path)
    net = lambda rows: stats.as_array(rows, "net_W2")
    c1 = stats.c1_policy_matters(net(arm("none", arm="no_update")), net(arm("oracle", arm="oracle")), blocks, counts)
    assert c1["pass"] and c1["point"] >= 0.05
    fixed = {d: net(arm(f"d{d}", arm="fixed", mode="E", fixed_d=d, d_a1=0.01)) for d in (0.1, 1.0, 5.0)}
    c2 = stats.c2_latency_matters(fixed, blocks, counts)
    assert c2["pass"] and c2["monotone"]


def test_controls_fail_and_h1_unresolved_when_radio_ignores_policy(tmp_path):
    s, blocks, counts, arm = _base_point(tmp_path, responsive=False)
    net = lambda rows: stats.as_array(rows, "net_W2")
    assert not stats.c1_policy_matters(net(arm("none", arm="no_update")), net(arm("oracle", arm="oracle")),
                                       blocks, counts)["pass"]
    arrays = {}
    for m, lat in [("Jev-1.13.0", 0.1), ("DeepSeek-V4.1-Flash", 3.0)]:
        rows = arm(m, arm="L", mode="E", records=records_for(s, m, lat), d_a1=0.01)
        arrays[m] = {"aff": stats.as_array(rows, "aff_W2"), "net": stats.as_array(rows, "net_W2")}
    res = stats.h1(arrays, blocks, counts)
    assert res["holds"]["DeepSeek-V4.1-Flash vs Jev-1.13.0"] is False
    assert all(t["point"] == 0.0 for t in res["tests"])


def test_h1_resolves_when_latency_hurts(tmp_path):
    s, blocks, counts, arm = _base_point(tmp_path)
    arrays = {}
    for m, lat in [("Jev-1.13.0", 0.1), ("DeepSeek-V4.1-Flash", 1.5), ("GLM-5.3-Flash", 0.1),
                   ("SemIf-Qwen3.5-4B", 0.1), ("Qwen3.5-4B-JSON", 0.6)]:
        rows = arm(m, arm="L", mode="E", records=records_for(s, m, lat), d_a1=0.01)
        arrays[m] = {"aff": stats.as_array(rows, "aff_W2"), "net": stats.as_array(rows, "net_W2")}
    res = stats.h1(arrays, blocks, counts)
    assert res["holds"]["DeepSeek-V4.1-Flash vs Jev-1.13.0"] is True
    assert res["holds"]["Qwen3.5-4B-JSON vs SemIf-Qwen3.5-4B"] is True
    assert res["holds"]["GLM-5.3-Flash vs Jev-1.13.0"] is False  # same latency: gap 0
    assert res["holds"]["Qwen3.8-Flash vs Jev-1.13.0"] is None    # arm missing
    assert {t["family_size"] for t in res["tests"]} == {6}
    # paired: identical arrays give an exactly zero bootstrap distribution
    g = stats.gap(arrays["Jev-1.13.0"]["aff"], arrays["GLM-5.3-Flash"]["aff"], blocks, counts)
    assert g["point"] == 0.0 and np.all(g["boot"] == 0.0)


def test_h1_needs_both_co_primary_metrics():
    rng = np.random.default_rng(2)
    n, k = 100, 10
    blocks, counts = np.repeat(np.arange(k), n // k), stats.draws(k)
    base_aff, base_net = rng.uniform(0.2, 0.4, n), rng.uniform(0.1, 0.2, n)
    arrays = {
        "Jev-1.13.0": {"aff": base_aff, "net": base_net},
        # affected-class gap large and resolved; network-wide gap exactly zero
        "DeepSeek-V4.1-Flash": {"aff": base_aff + 0.2 + rng.normal(0, 0.01, n), "net": base_net.copy()},
    }
    res = stats.h1(arrays, blocks, counts)
    by_metric = {t["metric"]: t for t in res["tests"]}
    assert by_metric["aff"]["resolved_pos"] and not by_metric["net"]["resolved_pos"]
    assert res["holds"]["DeepSeek-V4.1-Flash vs Jev-1.13.0"] is False
    arrays["DeepSeek-V4.1-Flash"]["net"] = base_net + 0.1 + rng.normal(0, 0.01, n)  # now both resolve
    assert stats.h1(arrays, blocks, counts)["holds"]["DeepSeek-V4.1-Flash vs Jev-1.13.0"] is True


def _fake_point(eligible: bool, gap_value: float, rng: np.random.Generator) -> dict:
    n, k = 100, 10
    blocks = np.repeat(np.arange(k), n // k)
    base = rng.uniform(0.2, 0.4, n)
    return {"eligible": eligible, "blocks": blocks, "counts": stats.draws(k),
            "aff": {"Jev-1.13.0": base, "DeepSeek-V4.1-Flash": base + gap_value + rng.normal(0, 0.01, n)}}


def test_h2_rules():
    rng = np.random.default_rng(0)
    pts = {f"p{i}": _fake_point(True, 0.1, rng) for i in range(9)}
    assert stats.h2(pts)["holds"]["DeepSeek-V4.1-Flash vs Jev-1.13.0"]["holds"]
    pts["p0"] = _fake_point(True, -0.1, rng)  # one reversed-and-resolved point kills it
    v = stats.h2(pts)["holds"]["DeepSeek-V4.1-Flash vs Jev-1.13.0"]
    assert v["n_reversed_resolved"] == 1 and not v["holds"]
    few = {f"p{i}": _fake_point(i < 4, 0.1, rng) for i in range(9)}  # only 4 eligible
    assert not stats.h2(few)["holds"]["DeepSeek-V4.1-Flash vs Jev-1.13.0"]["holds"]
    mixed = {f"p{i}": _fake_point(True, 0.1 if i < 7 else 0.0, rng) for i in range(9)}  # 7/9 < 80 %
    assert not stats.h2(mixed)["holds"]["DeepSeek-V4.1-Flash vs Jev-1.13.0"]["holds"]


def test_block_ids_draws_rho_and_block_length():
    b = stats.block_ids(np.array([2.0, 11.99, 12.0, 335.0]), 2.0, 10.0, 33)
    assert list(b) == [0, 0, 1, 32]
    assert np.array_equal(stats.draws(12), stats.draws(12))  # same boundaries + seed -> same draws for every arm
    assert stats.utilisation(1.0, np.array([4.0, 4.4]), 4)["non_stationary"]
    assert not stats.utilisation(0.5, np.array([4.0]), 4)["non_stationary"]
    rng = np.random.default_rng(1)
    white = rng.normal(size=20000)
    ar = np.zeros(20000)
    for i in range(1, len(ar)):
        ar[i] = 0.99 * ar[i - 1] + rng.normal()
    assert stats.autocorr_time_s(white, 0.1) < 0.2
    assert stats.autocorr_time_s(ar, 0.1) > 10.0  # tau ~ (1 + 0.99) / (1 - 0.99) = 199 slots
    assert stats.block_length_s(white, 0.1) == 10.0


def test_network_series(tmp_path):
    s = _one_intent_stream()
    tr = load_run(write_run(tmp_path / "n", s, []))
    ts, share = network_series(tr)
    assert len(ts) == 250 and share[ts < 10.05].max() == 0.0 and share[ts > 10.1].min() == pytest.approx(2 / 6)


# ------------------------------------------------------------------ smoke on the real ns-3 check runs

REAL = Path(__file__).resolve().parents[1] / "c2-check-runs"  # ns-3 check runs; not distributed


@pytest.mark.skipif(not (REAL / "check1a" / "ue_slots.csv").exists(), reason="ns-3 check runs not present")
@pytest.mark.parametrize("run", ["check1a", "check2-run1"])
def test_smoke_real_check_runs(run):
    tr = load_run(REAL / run, tolerate_torn_tail=True)
    with open(REAL / run / "schedule.csv") as f:
        sched = list(csv.DictReader(f))
    t_end = float(tr.ue_slots["t"].max())
    events = [{"j": j, "intent_id": f"s{j}", "t_issue_s": float(r["t_s"]), "e_s": float(r["t_s"]), "correct": True,
               "true_install": [r["scope"], r["class"], r["priority"]],
               "install": [r["scope"], r["class"], r["priority"]]} for j, r in enumerate(sched)]
    rows = intent_outcomes(tr, events, t_end=t_end)
    kpis = radio_kpis(tr, 0.2, t_end)
    assert len(rows) == len(sched) and kpis["n_ue_slots"] > 0
    for r in rows:
        if r["t_issue_s"] + 2 <= t_end:
            assert 0.0 <= r["net_W2"] <= 1.0


def test_c4_pairing_ignores_serving_cell_but_not_positions(tmp_path):
    from src.ranbench.c2.loader import pairing_identical

    def write(d, rows):
        d.mkdir()
        (d / "positions.csv").write_text("t,imsi,x,y,serving_cell\n" + "".join(r + "\n" for r in rows))
        (d / "tx_trace.csv").write_text("t_slot,imsi,n_pkts,bytes\n0.1,1,3,3900\n")

    base = ["0.1,1,10.0,20.0,4", "0.2,1,10.5,20.1,4"]
    write(tmp_path / "a", base)
    write(tmp_path / "b", ["0.1,1,10.0,20.0,4", "0.2,1,10.5,20.1,5"])  # handover shifted across a sample
    write(tmp_path / "c", ["0.1,1,10.0,20.0,4", "0.2,1,10.6,20.1,4"])  # a position differs
    assert pairing_identical(tmp_path / "a", tmp_path / "b") == {"positions.csv": True, "tx_trace.csv": True}
    assert pairing_identical(tmp_path / "a", tmp_path / "c")["positions.csv"] is False
