"""Tests for the C2 return path: rb_c2_verify_runs.py, rb_c2_pack_zip.py, rb_c2_ingest_zip.py.

The scripts are run as subprocesses, exactly as the handover runs them.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
VERIFY = REPO / "scripts" / "rb_c2_verify_runs.py"
PACK = REPO / "scripts" / "rb_c2_pack_zip.py"
INGEST = REPO / "scripts" / "rb_c2_ingest_zip.py"
LENA = "cedceadda17392c90587fb9400eb9b1f8c236713"
BINARY_SHA = "b" * 64
M4_INPUTS = "/data/c2-production-inputs"  # matrix paths are relocated

HEADERS = {
    "ue_slots.csv": "t_slot,imsi,serving_cell,class,rx_bytes,n_rx_pkts,mean_delay_ms,p95_delay_ms",
    "tx_trace.csv": "t_slot,imsi,tx_bytes,n_tx_pkts",
    "positions.csv": "t,imsi,x,y,serving_cell",
    "cell_prb.csv": "t_slot,cell,prb_share",
    "handover.csv": "t_start,t_end_ok,imsi,source_cell,target_cell,handover_total_time_ms,outcome",
    "rlf.csv": "t,imsi,cell,cause",
    "enforcement.csv": "t,cell,class,old,new,source",
    "lc_priority.csv": "t,cell,rnti,imsi,class,lc_m_priority,cell_class_level",
    "cells.csv": "cell,site,sector,x,y,bearing_deg",
    "ue_map.csv": "ue_index,imsi,drop_cell,class,port",
}
CELLS, UES, SIM_S = 2, 3, 0.3  # 3 report slots


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_inputs(root: Path, run_id: str) -> None:
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "schedules").mkdir(exist_ok=True)
    (root / "streams").mkdir(exist_ok=True)
    (root / "configs" / "speed_3_ues_1.json").write_text(
        json.dumps({"scenario": {"num_cells": CELLS}}) + "\n", encoding="utf-8")
    (root / "schedules" / f"{run_id}.csv").write_text(
        "t_s,scope,class,priority\n2.500000,all_cells,xr,high\n", encoding="utf-8")
    (root / "streams" / "rate_0p1_speed_3.json").write_text('{"events": []}\n', encoding="utf-8")


def make_run(run_root: Path, inputs: Path, run_id: str, binary_sha: str = BINARY_SHA) -> Path:
    run = run_root / run_id
    run.mkdir(parents=True)
    copies = {"config.json": inputs / "configs" / "speed_3_ues_1.json",
              "schedule.csv": inputs / "schedules" / f"{run_id}.csv",
              "stream.json": inputs / "streams" / "rate_0p1_speed_3.json"}
    for name, src in copies.items():
        (run / name).write_bytes(src.read_bytes())
    slots = [f"{0.1 * k:.3f}" for k in range(1, 4)]
    body = {
        "ue_slots.csv": [f"{t},{i},1,video,100,1,1.0,2.0" for t in slots for i in range(UES)],
        "tx_trace.csv": [f"{t},{i},100,1" for t in slots for i in range(UES)],
        "positions.csv": [f"{t},{i},0.0,0.0,1" for t in slots for i in range(UES)],
        "cell_prb.csv": [f"{t},{c},0.5" for t in slots for c in range(1, CELLS + 1)],
        "handover.csv": [],
        "rlf.csv": [],
        "enforcement.csv": ["1.000000,1,video,50,48,xapp"],
        "lc_priority.csv": [f"{t},1,1,0,video,50,50" for t in slots],
        "cells.csv": [f"{c},0,{c - 1},0.0,0.0,30" for c in range(1, CELLS + 1)],
        "ue_map.csv": [f"{i},{i},1,video,{12000 + i}" for i in range(UES)],
    }
    for name, rows in body.items():
        (run / name).write_text("\n".join([HEADERS[name], *rows]) + "\n", encoding="utf-8")
    (run / "streams.json").write_text(json.dumps({"rng_run": 1, "rng_seed": 1}), encoding="utf-8")
    (run / "stdout.log").write_text("", encoding="utf-8")
    (run / "stderr.log").write_text("real 1.0\n", encoding="utf-8")
    manifest = {
        "ns3_version": "ns-3.48", "lena_commit": LENA, "build_profile": "optimized",
        "binary_sha256": binary_sha, "config_sha256": sha(run / "config.json"),
        "schedule_sha256": sha(run / "schedule.csv"), "stream_sha256": sha(run / "stream.json"),
        "rng_run": 1, "sim_time_s": SIM_S, "status": "complete", "complete": True,
    }
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return run


def setup(tmp_path: Path, run_ids=("r1",)) -> dict[str, Path]:
    inputs, runs = tmp_path / "inputs", tmp_path / "runs"
    lines = ["run_id,priority_tier,ues,simulated_seconds,config_file,schedule_file,stream_file,rng_run"]
    for run_id in run_ids:
        make_inputs(inputs, run_id)
        make_run(runs, inputs, run_id)
        lines.append(f"{run_id},1,{UES},{SIM_S},{M4_INPUTS}/configs/speed_3_ues_1.json,"
                     f"{M4_INPUTS}/schedules/{run_id}.csv,{M4_INPUTS}/streams/rate_0p1_speed_3.json,1")
    (tmp_path / "matrix.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (tmp_path / "runlist.txt").write_text("# M5\n" + "\n".join(run_ids) + "\n", encoding="utf-8")
    return {"inputs": inputs, "runs": runs, "matrix": tmp_path / "matrix.csv",
            "runlist": tmp_path / "runlist.txt"}


def common(p: dict[str, Path]) -> list[str]:
    return ["--run-list", str(p["runlist"]), "--matrix", str(p["matrix"]),
            "--input-root", str(p["inputs"])]


def run(script: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True,
                          check=False)


def verify(p: dict[str, Path], *extra: str) -> subprocess.CompletedProcess:
    return run(VERIFY, "--run-root", str(p["runs"]), *common(p), *extra)


# ---------------------------------------------------------------- verify


def test_verify_passes_well_formed_run(tmp_path):
    p = setup(tmp_path, ("r1", "r2"))
    proc = verify(p, "--binary-sha256", BINARY_SHA)
    assert proc.returncode == 0, proc.stdout
    assert "SUMMARY PASS=2" in proc.stdout and "RESULT PASS" in proc.stdout


def test_verify_rejects_wrong_header(tmp_path):
    p = setup(tmp_path)
    path = p["runs"] / "r1" / "tx_trace.csv"
    path.write_text(path.read_text().replace("n_tx_pkts", "n_pkts"), encoding="utf-8")
    proc = verify(p)
    assert proc.returncode == 1 and "tx_trace.csv header" in proc.stdout


def test_verify_rejects_truncated_report(tmp_path):
    p = setup(tmp_path)
    path = p["runs"] / "r1" / "ue_slots.csv"
    path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n", encoding="utf-8")
    proc = verify(p)
    assert proc.returncode == 1 and "ue_slots.csv has 8 data rows, expected 9" in proc.stdout


def test_verify_rejects_input_hash_mismatch(tmp_path):
    p = setup(tmp_path)
    (p["inputs"] / "schedules" / "r1.csv").write_text("t_s,scope,class,priority\n", encoding="utf-8")
    proc = verify(p)
    assert proc.returncode == 1
    assert "manifest schedule_sha256 != sha256 of r1.csv" in proc.stdout


def test_verify_rejects_manifest_facts(tmp_path):
    p = setup(tmp_path)
    path = p["runs"] / "r1" / "manifest.json"
    data = json.loads(path.read_text())
    data.update(lena_commit="deadbeef", sim_time_s=0.4, complete=False)
    path.write_text(json.dumps(data), encoding="utf-8")
    proc = verify(p)
    assert proc.returncode == 1
    for text in ("lena_commit", "sim_time_s", "manifest not complete"):
        assert text in proc.stdout


@pytest.mark.parametrize("version, ok", [("ns-3.48-d7", True), ("ns-3.48-d7b", True), ("ns-3.47", False)])
def test_verify_ns3_version(tmp_path, version, ok):
    """D-7/D-7b/D-11 rerun builds pass; any other ns-3 tree fails."""
    p = setup(tmp_path)
    path = p["runs"] / "r1" / "manifest.json"
    data = json.loads(path.read_text())
    data.update(ns3_version=version)
    path.write_text(json.dumps(data), encoding="utf-8")
    proc = verify(p)
    assert (proc.returncode == 0) is ok, proc.stdout
    assert ("ns3_version" in proc.stdout) is not ok


def test_verify_binary_checks(tmp_path):
    p = setup(tmp_path, ("r1", "r2"))
    assert "binary_sha256" in verify(p, "--binary-sha256", "a" * 64).stdout
    path = p["runs"] / "r2" / "manifest.json"
    data = json.loads(path.read_text())
    data["binary_sha256"] = "c" * 64
    path.write_text(json.dumps(data), encoding="utf-8")
    proc = verify(p)
    assert proc.returncode == 1 and "runs used different binaries" in proc.stdout


def test_verify_missing_and_incomplete(tmp_path):
    p = setup(tmp_path, ("r1", "r2", "r3"))
    (p["runs"] / "r2" / "manifest.json").unlink()
    for child in (p["runs"] / "r3").iterdir():
        child.unlink()
    (p["runs"] / "r3").rmdir()
    proc = verify(p)
    assert proc.returncode == 1 and "INCOMPLETE=1" in proc.stdout and "MISSING=1" in proc.stdout
    assert verify(p, "--allow-missing").returncode == 0


# ---------------------------------------------------------------- pack + ingest


def pack(p: dict[str, Path], outbox: Path) -> subprocess.CompletedProcess:
    return run(PACK, "--run-root", str(p["runs"]), *common(p), "--outbox", str(outbox))


def ingest(p: dict[str, Path], dest: Path, *zips: Path) -> subprocess.CompletedProcess:
    return run(INGEST, *map(str, zips), "--dest", str(dest), *common(p))


def only_zip(outbox: Path) -> Path:
    zips = sorted(outbox.glob("*.zip"))
    assert len(zips) == 1
    return zips[0]


def test_round_trip_is_byte_identical(tmp_path):
    p = setup(tmp_path, ("r1", "r2"))
    outbox, dest = tmp_path / "outbox", tmp_path / "dest"
    proc = pack(p, outbox)
    assert proc.returncode == 0, proc.stdout
    zip_path = only_zip(outbox)
    assert "_EXP-2026-003_c2_b01_" in zip_path.name
    assert "runs: 2" in (outbox / (zip_path.name + ".manifest.txt")).read_text()
    assert "NOTHING TO PACK" in pack(p, outbox).stdout  # already shipped
    proc = ingest(p, dest, zip_path)
    assert proc.returncode == 0, proc.stdout
    for run_id in ("r1", "r2"):
        for src in (p["runs"] / run_id).iterdir():
            assert (dest / run_id / src.name).read_bytes() == src.read_bytes()
    assert not list(dest.glob(".ingest-*"))


def test_pack_refuses_failing_run(tmp_path):
    p = setup(tmp_path)
    (p["runs"] / "r1" / "rlf.csv").unlink()
    proc = pack(p, tmp_path / "outbox")
    assert proc.returncode == 1 and "REFUSED" in proc.stdout
    assert not list((tmp_path / "outbox").glob("*.zip"))


def test_ingest_rejects_sha_mismatch(tmp_path):
    p = setup(tmp_path)
    outbox, dest = tmp_path / "outbox", tmp_path / "dest"
    pack(p, outbox)
    zip_path = only_zip(outbox)
    (outbox / (zip_path.name + ".sha256")).write_text(f"{'0' * 64}  {zip_path.name}\n")
    proc = ingest(p, dest, zip_path)
    assert proc.returncode == 1 and "ZIP_REJECTED" in proc.stdout
    assert not (dest / "r1").exists()


def test_ingest_never_overwrites(tmp_path):
    p = setup(tmp_path, ("r1", "r2"))
    outbox, dest = tmp_path / "outbox", tmp_path / "dest"
    pack(p, outbox)
    zip_path = only_zip(outbox)
    make_inputs(tmp_path / "other", "r1")
    existing = make_run(dest, tmp_path / "other", "r1")
    (existing / "marker.txt").write_text("keep me", encoding="utf-8")
    (dest / "r2").mkdir()  # partial directory, no manifest
    proc = ingest(p, dest, zip_path)
    assert proc.returncode == 1
    assert "SKIP_EXISTING" in proc.stdout and "REFUSE_EXISTING" in proc.stdout
    assert (existing / "marker.txt").read_text() == "keep me"
    assert list((dest / "r2").iterdir()) == []


def test_ingest_leaves_failing_run_in_staging(tmp_path):
    p = setup(tmp_path)
    outbox, dest = tmp_path / "outbox", tmp_path / "dest"
    pack(p, outbox)
    zip_path = only_zip(outbox)
    # Change the frozen input on the receiving side: the run no longer matches it.
    (p["inputs"] / "configs" / "speed_3_ues_1.json").write_text('{"scenario": {"num_cells": 3}}\n')
    proc = ingest(p, dest, zip_path)
    assert proc.returncode == 1 and "FAIL" in proc.stdout
    assert not (dest / "r1").exists()
    assert (dest / f".ingest-{zip_path.stem}" / "r1" / "manifest.json").is_file()


def test_ingest_rejects_unlisted_or_unsafe_members(tmp_path):
    p = setup(tmp_path)
    outbox, dest = tmp_path / "outbox", tmp_path / "dest"
    outbox.mkdir()
    zip_path = outbox / "m5_EXP-2026-003_c2_b01_20260101.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("../escape.txt", "x")
        zf.writestr("stranger/manifest.json", "{}")
    (outbox / (zip_path.name + ".sha256")).write_text(f"{sha(zip_path)}  {zip_path.name}\n")
    (outbox / (zip_path.name + ".manifest.txt")).write_text("run: stranger files=1 bytes=2\n")
    proc = ingest(p, dest, zip_path)
    assert proc.returncode == 1
    assert "unexpected member '../escape.txt'" in proc.stdout
    assert "stranger is not in the run list" in proc.stdout
    assert not (tmp_path / "escape.txt").exists() and not (dest / "stranger").exists()


def test_quarantine_moves_only_unfinished_runs(tmp_path):
    p = setup(tmp_path, ("r1", "r2"))
    (p["runs"] / "r2" / "manifest.json").unlink()
    proc = verify(p, "--quarantine")
    assert proc.returncode == 0 and "QUARANTINED r2" in proc.stdout
    assert (p["runs"] / "r1" / "manifest.json").is_file() and not (p["runs"] / "r2").exists()
    moved = list((p["runs"] / "_incomplete").iterdir())
    assert len(moved) == 1 and moved[0].name.startswith("r2.")
    assert verify(p).stdout.count("MISSING") == 2  # r2 is now absent, so the queue can rerun it


# ---------------------------------------------------------------- D-2 RQ5 live runs


def _live_run(tmp_path: Path, rqs: str) -> tuple[Path, dict[str, str], Path]:
    """An RQ5-harness style run: live schedule (no input file), empty stderr.log, manifest mode=live."""
    from src.ranbench.c2.verify import read_matrix
    p = setup(tmp_path)
    run_dir = p["runs"] / "r1"
    (p["inputs"] / "schedules" / "r1.csv").unlink()
    (run_dir / "stderr.log").write_text("", encoding="utf-8")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest["mode"] = "live"
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    row = dict(read_matrix(p["matrix"])["r1"])
    row["rqs"] = rqs
    return run_dir, row, p["inputs"]


def test_verify_rq5_live_run_passes(tmp_path):
    from src.ranbench.c2.verify import verify_run
    run_dir, row, inputs = _live_run(tmp_path, "RQ5")
    assert verify_run(run_dir, row, inputs) == ("PASS", [])


def test_verify_live_exceptions_need_an_rq5_row(tmp_path):
    from src.ranbench.c2.verify import verify_run
    run_dir, row, inputs = _live_run(tmp_path, "RQ2")
    status, problems = verify_run(run_dir, row, inputs)
    assert status == "FAIL"
    assert "stderr.log missing or empty" in problems
    assert any(p.startswith("input ") and "r1.csv not found" in p for p in problems)


def test_verify_rq5_live_schedule_bound_to_manifest(tmp_path):
    from src.ranbench.c2.verify import verify_run
    run_dir, row, inputs = _live_run(tmp_path, "RQ5")
    (run_dir / "schedule.csv").write_text("t_s,scope,class,priority\n", encoding="utf-8")
    status, problems = verify_run(run_dir, row, inputs)
    assert status == "FAIL" and "live schedule schedule.csv does not match manifest schedule_sha256" in problems


def test_verify_rq5_live_stderr_must_exist(tmp_path):
    from src.ranbench.c2.verify import verify_run
    run_dir, row, inputs = _live_run(tmp_path, "RQ5")
    (run_dir / "stderr.log").unlink()
    status, problems = verify_run(run_dir, row, inputs)
    assert status == "FAIL" and "stderr.log missing" in problems
