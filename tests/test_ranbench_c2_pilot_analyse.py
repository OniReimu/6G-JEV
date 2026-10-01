from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from src.ranbench.c2.pilot import PILOT_ARMS, analyse, prepare

REPO = Path(__file__).resolve().parents[1]
UES = ((1, 1, "video"), (2, 7, "xr"), (3, 13, "iot"), (4, 4, "be"))  # imsi, cell, class
SIM_S = 20.0


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_run(pilot: Path, run: Path, arm: str) -> None:
    """Synthetic traces: IoT UE 3 silent over (5, 8] after an RLF in every arm; no-update starves video."""
    run.mkdir(parents=True)
    shutil.copy(pilot / "config.json", run / "config.json")
    shutil.copy(pilot / "arms" / arm / "schedule.csv", run / "schedule.csv")
    slots = ["t_slot,imsi,serving_cell,class,rx_bytes,n_rx_pkts,mean_delay_ms,p95_delay_ms"]
    tx = ["t_slot,imsi,tx_bytes,n_tx_pkts"]
    prb = ["t_slot,cell,prb_share"]
    for k in range(1, int(SIM_S * 10) + 1):
        t = f"{k / 10:.3f}"
        for imsi, cell, cls in UES:
            rx = 20000
            if cls == "video" and arm == "no-update":
                rx = 0
            if cls == "iot" and 5.0 < k / 10 <= 8.0 + 1e-9:
                rx = 0
            slots.append(f"{t},{imsi},{cell},{cls},{rx},{10 if rx else 0},{'3.0' if rx else ''},{'5.0' if rx else ''}")
            tx.append(f"{t},{imsi},20000,10")
            prb.append(f"{t},{cell},{0.5 if cell != 1 else 0.97}")
    (run / "ue_slots.csv").write_text("\n".join(slots) + "\n")
    (run / "tx_trace.csv").write_text("\n".join(tx) + "\n")
    (run / "cell_prb.csv").write_text("\n".join(prb) + "\n")
    (run / "handover.csv").write_text(
        "t_start,t_end_ok,imsi,source_cell,target_cell,handover_total_time_ms,outcome\n")
    (run / "rlf.csv").write_text("t,imsi,cell,cause\n5.0,3,13,CONNECTION_TIMEOUT\n")
    (run / "enforcement.csv").write_text("t,cell,class,old,new,source\n")
    (run / "manifest.json").write_text(json.dumps({
        "complete": True, "sim_time_s": SIM_S, "config_sha256": _sha(run / "config.json"),
        "schedule_sha256": _sha(run / "schedule.csv")}))


@pytest.fixture()
def candidate(tmp_path: Path) -> tuple[Path, Path]:
    pilot, runs = tmp_path / "pilot", tmp_path / "runs"
    prepare(pilot, REPO / "src/ranbench/ns3/c2-default.json",
            REPO / "data/ranbench/ranintent-v1/pool/c2c3_pool.jsonl", "balanced-c", SIM_S, 0.004300475120544434)
    for arm in PILOT_ARMS:
        _write_run(pilot, runs / arm, arm)
    return pilot, runs


def test_analyse_reports_controls_and_diagnostics(candidate: tuple[Path, Path]) -> None:
    pilot, runs = candidate
    r = analyse(pilot, runs)
    assert r["status"] == "complete" and r["horizon_s"] == SIM_S and r["n_events"] > 0
    saved = json.loads((pilot / "analysis.json").read_text())
    assert saved["controls"]["C-1-network-descriptive"]["point"] == pytest.approx(0.25)
    ue_slots_post = 180  # slots ending in (2, 20]
    assert r["sla_violation_share_by_class"]["oracle"] == pytest.approx(
        {"network": 30 / (4 * ue_slots_post), "video": 0.0, "xr": 0.0, "iot": 30 / ue_slots_post})
    assert r["sla_violation_share_by_class"]["no-update"]["video"] == pytest.approx(1.0)
    assert r["controls"]["C-2-network-descriptive"]["point"] == pytest.approx(0.0)
    assert r["controls"]["C-2-network-descriptive"]["pass"] is False
    act = r["actuation"]["intents"]
    assert r["controls"]["C-1"]["n_actuating"] == sum(a["actuating"] for a in act)
    assert all(a["cls"] in ("video", "xr", "iot") for a in act if a["actuating"])
    assert r["actuation"]["oracle_policy_rows_check"] == {"n_policy_rows": 0, "n_mismatch": 0}
    assert r["iot_responsiveness"]["rate_kbps"] == 300.0
    assert r["prb_utilisation"]["oracle"]["n_cells_ge_0p95"] == 1
    assert r["silent_periods"]["oracle"] == [[3, 5.0, 8.0]]
    assert r["silent_periods_identical"] is True
    assert r["silent_share"]["oracle"]["ho_rlf_associated"] == pytest.approx(3.0 / (4 * 18.0))
    assert r["block_length_s"] >= 10.0 and all(v > 0 for v in r["autocorr_time_by_arm_s"].values())
    assert r["warnings"]  # a 20-s run has one block


def test_analyse_fails_closed_on_missing_or_mismatched_manifest(candidate: tuple[Path, Path]) -> None:
    pilot, runs = candidate
    (runs / "fixed-5" / "schedule.csv").write_text("t_s,scope,class,priority\n")
    m = json.loads((runs / "fixed-5" / "manifest.json").read_text())
    m["schedule_sha256"] = _sha(runs / "fixed-5" / "schedule.csv")
    (runs / "fixed-5" / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(RuntimeError, match="fixed-5: schedule differs"):
        analyse(pilot, runs)
    (runs / "oracle" / "manifest.json").unlink()
    with pytest.raises(RuntimeError, match="oracle: .*manifest.json missing"):
        analyse(pilot, runs)


def test_partial_look_uses_common_horizon(candidate: tuple[Path, Path]) -> None:
    pilot, runs = candidate
    for arm in PILOT_ARMS:
        (runs / arm / "manifest.json").unlink()
    path = runs / "no-update" / "ue_slots.csv"
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[: 1 + 4 * 120]) + "\n12.1,1,1,vid")  # through t = 12.0, torn tail
    r = analyse(pilot, runs, allow_partial=True)
    assert r["status"].startswith("PARTIAL") and r["horizon_s"] == pytest.approx(12.0)
    assert (pilot / "analysis.partial.json").is_file() and not (pilot / "analysis.json").exists()
