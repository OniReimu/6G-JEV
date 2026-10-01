from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from src.ranbench.c2.eligibility import eligibility_flags, evaluate, main
from src.ranbench.c2.pilot import PILOT_ARMS, analyse, prepare
from src.ranbench.c2.verify import LENA_COMMIT, NS3_VERSION

REPO = Path(__file__).resolve().parents[1]
UES = ((1, 1, "video"), (2, 7, "xr"), (3, 13, "iot"), (4, 4, "be"))
SIM_S = 20.0
BINARY_SHA = "b" * 64
DESIGN_POINT = "synthetic-point"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_run(pilot: Path, run: Path, arm: str) -> None:
    run.mkdir(parents=True)
    shutil.copy(pilot / "config.json", run / "config.json")
    shutil.copy(pilot / "stream.json", run / "stream.json")
    shutil.copy(pilot / "arms" / arm / "schedule.csv", run / "schedule.csv")
    slots = ["t_slot,imsi,serving_cell,class,rx_bytes,n_rx_pkts,mean_delay_ms,p95_delay_ms"]
    tx = ["t_slot,imsi,tx_bytes,n_tx_pkts"]
    positions = ["t,imsi,x,y,serving_cell"]
    prb = ["t_slot,cell,prb_share"]
    lc = ["t,cell,rnti,imsi,class,lc_m_priority,cell_class_level"]
    for k in range(1, int(SIM_S * 10) + 1):
        t = f"{k / 10:.3f}"
        for imsi, cell, cls in UES:
            rx = 20000
            if cls == "video" and arm == "no-update":
                rx = 0
            if cls == "iot" and 5.0 < k / 10 <= 8.0 + 1e-9:
                rx = 0
            slots.append(f"{t},{imsi},{cell},{cls},{rx},{10 if rx else 0},"
                         f"{'3.0' if rx else ''},{'5.0' if rx else ''}")
            tx.append(f"{t},{imsi},20000,10")
            positions.append(f"{t},{imsi},0.0,0.0,{cell}")
        for cell in range(1, 22):
            prb.append(f"{t},{cell},{0.97 if cell == 1 else 0.5}")
    lc.append("0.100,1,1,1,video,50,50")
    tables = {
        "ue_slots.csv": slots,
        "tx_trace.csv": tx,
        "positions.csv": positions,
        "cell_prb.csv": prb,
        "handover.csv": [
            "t_start,t_end_ok,imsi,source_cell,target_cell,handover_total_time_ms,outcome"],
        "rlf.csv": ["t,imsi,cell,cause", "5.0,3,13,CONNECTION_TIMEOUT"],
        "enforcement.csv": ["t,cell,class,old,new,source"],
        "lc_priority.csv": lc,
        "cells.csv": ["cell,site,sector,x,y,bearing_deg"]
                     + [f"{cell},0,0,0.0,0.0,0.0" for cell in range(1, 22)],
        "ue_map.csv": ["ue_index,imsi,drop_cell,class,port"]
                      + [f"{index},{imsi},{cell},{cls},{12000 + index}"
                         for index, (imsi, cell, cls) in enumerate(UES)],
    }
    for name, lines in tables.items():
        (run / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run / "streams.json").write_text('{"rng_run": 1}\n', encoding="utf-8")
    (run / "stdout.log").write_text("", encoding="utf-8")
    (run / "stderr.log").write_text("real 1.0\n", encoding="utf-8")
    (run / "manifest.json").write_text(json.dumps({
        "complete": True,
        "status": "complete",
        "sim_time_s": SIM_S,
        "rng_run": 1,
        "ns3_version": NS3_VERSION,
        "lena_commit": LENA_COMMIT,
        "build_profile": "optimized",
        "binary_sha256": BINARY_SHA,
        "config_sha256": _sha(run / "config.json"),
        "schedule_sha256": _sha(run / "schedule.csv"),
        "stream_sha256": _sha(run / "stream.json"),
    }), encoding="utf-8")


@pytest.fixture()
def design_point(tmp_path: Path) -> tuple[Path, Path, Path]:
    pilot, runs, inputs = tmp_path / "pilot", tmp_path / "runs", tmp_path / "inputs"
    prepare(pilot, REPO / "src/ranbench/ns3/c2-default.json",
            REPO / "data/ranbench/ranintent-v1/pool/c2c3_pool.jsonl",
            "balanced-c", SIM_S, 0.004300475120544434)
    for name in ("configs", "schedules", "streams"):
        (inputs / name).mkdir(parents=True)
    config = inputs / "configs" / "base.json"
    stream = inputs / "streams" / "base.json"
    shutil.copy(pilot / "config.json", config)
    shutil.copy(pilot / "stream.json", stream)
    rows = []
    for arm in PILOT_ARMS:
        _write_run(pilot, runs / arm, arm)
        schedule = inputs / "schedules" / f"{arm}.csv"
        shutil.copy(pilot / "arms" / arm / "schedule.csv", schedule)
        shutil.copy(pilot / "arms" / arm / "events.json", schedule.with_suffix(".events.json"))
        rows.append({
            "run_id": arm,
            "design_point": DESIGN_POINT,
            "arm": arm,
            "mode": "E",
            "interpreter": "control",
            "ues": str(len(UES)),
            "simulated_seconds": str(SIM_S),
            "config_file": str(config),
            "schedule_file": str(schedule),
            "stream_file": str(stream),
        })
    matrix = tmp_path / "matrix.csv"
    with matrix.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return pilot, runs, matrix


def test_production_eligibility_matches_pilot_numbers_exactly(
        design_point: tuple[Path, Path, Path]) -> None:
    pilot, runs, matrix = design_point
    pilot_result = analyse(pilot, runs)
    result = evaluate(matrix, [runs], DESIGN_POINT)
    assert result["rule"] == "D-10"
    assert isinstance(result["eligible"], bool)
    assert isinstance(result["eligible_10s"], bool)
    assert result["eligible_d10"] == (result["eligible"] and result["eligible_10s"])
    assert isinstance(result["eligible_original_5pp"], bool)
    assert result["controls"] == pilot_result["controls"]
    assert result["controls_10s"] == pilot_result["controls_10s"]
    assert result["autocorr_time_by_arm_s"] == pilot_result["autocorr_time_by_arm_s"]
    assert result["block_length_s"] == pilot_result["block_length_s"]
    assert result["block_length_by_arm_s"] == pilot_result["block_length_by_arm_s"]
    assert result["n_blocks"] == pilot_result["n_blocks"]
    assert result["n_blocks_10s"] == pilot_result["n_blocks_10s"]
    assert result["run_ids"] == {arm: arm for arm in PILOT_ARMS}
    assert result["binary_sha256_by_run"] == {arm: BINARY_SHA for arm in PILOT_ARMS}
    out = matrix.parent / "eligibility.json"
    main(["--matrix", str(matrix), "--runs-root", str(runs),
          "--design-point", DESIGN_POINT, "--out", str(out)])
    assert json.loads(out.read_text(encoding="utf-8")) == json.loads(json.dumps(result))


@pytest.mark.parametrize(
    "c1,c2,eligible,eligible_original_5pp",
    [
        ({"point": 0.025, "ci_low": 0.012, "ci_high": 0.041, "p_raw": 0.01, "pass": False},
         {"pass": True, "monotone": True}, True, False),
        ({"point": 0.025, "ci_low": -0.01, "ci_high": 0.06, "p_raw": 0.01, "pass": False},
         {"pass": True, "monotone": True}, False, False),
        ({"point": -0.025, "ci_low": -0.041, "ci_high": -0.012, "p_raw": 0.01, "pass": False},
         {"pass": True, "monotone": True}, False, False),
        ({"point": 0.06, "ci_low": 0.04, "ci_high": 0.08, "p_raw": 0.01, "pass": True},
         {"pass": False, "monotone": False}, False, False),
    ],
    ids=["positive-2.5pp-resolved", "positive-unresolved", "negative-resolved", "c2-non-monotone"],
)
def test_d9_eligibility_rules(c1: dict[str, float | bool], c2: dict[str, bool],
                              eligible: bool, eligible_original_5pp: bool) -> None:
    controls = {"C-1": c1, "C-2": c2}
    assert eligibility_flags(controls, block_pass=True) == {
        "eligible": eligible,
        "eligible_original_5pp": eligible_original_5pp,
    }


def test_incomplete_control_run_is_an_error(design_point: tuple[Path, Path, Path]) -> None:
    _, runs, matrix = design_point
    (runs / "fixed-5" / "manifest.json").unlink()
    with pytest.raises(RuntimeError, match="fixed-5.*INCOMPLETE"):
        evaluate(matrix, [runs], DESIGN_POINT)


def test_legacy_block_argument_is_informational(
        design_point: tuple[Path, Path, Path]) -> None:
    _, runs, matrix = design_point
    result = evaluate(matrix, [runs], DESIGN_POINT, block_s=99.0)
    assert result["block_check"]["requested_block_s"] == 99.0
    assert result["block_check"]["pass"] is True
    assert result["block_s"] == result["block_check"]["configured_block_s"]


def test_mixed_binary_sha_is_an_error(design_point: tuple[Path, Path, Path]) -> None:
    _, runs, matrix = design_point
    path = runs / "fixed-1" / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["binary_sha256"] = "c" * 64
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="different binary sha256"):
        evaluate(matrix, [runs], DESIGN_POINT)


def test_missing_control_is_an_error(design_point: tuple[Path, Path, Path]) -> None:
    _, runs, matrix = design_point
    with matrix.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["arm"] != "no-update"]
    with matrix.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="missing control no-update"):
        evaluate(matrix, [runs], DESIGN_POINT)
