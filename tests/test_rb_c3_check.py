from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "c3_check"
CHECKER = ROOT / "scripts" / "rb_c3_check.py"


def run_checker(run_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "--records", str(run_dir / "records.jsonl"),
            "--a1", str(run_dir / "records.a1.jsonl"),
            "--xapp", str(run_dir / "xapp.jsonl"),
            "--gnb", str(run_dir / "gnb.log"),
            "--iperf3", str(run_dir / "iperf3.json"),
            "--expected", "2",
            "--output", str(run_dir / "integrity.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_checker_passes_complete_fixture(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    shutil.copytree(FIXTURE, run_dir)
    result = run_checker(run_dir)
    report = json.loads((run_dir / "integrity.json").read_text())
    assert result.returncode == 0, result.stderr
    assert report["status"] == "PASS"
    assert report["actual_decisions"] == 2
    assert report["actuated_decisions"] == 2
    assert report["invalid_decisions"] == 0
    assert report["unactuatable_decisions"] == 0
    assert report["strict_gnb_exchanges"] == 2
    assert report["summary"]["ell_s"] == {
        "n": 2, "null_count": 0, "median_s": 0.30000000000000004, "min_s": 0.2, "max_s": 0.4,
    }
    assert report["summary"]["kpm_change_upper_bound_s"]["null_count"] == 1


def test_checker_accepts_operator_sigint_iperf3_error_with_complete_run_coverage(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    shutil.copytree(FIXTURE, run_dir)
    result = run_checker(run_dir)
    report = json.loads((run_dir / "integrity.json").read_text())
    assert result.returncode == 0, result.stderr
    assert report["iperf3_intervals"] == 1
    assert not any("iperf3 reported an error" in error for error in report["errors"])


def test_checker_rejects_other_iperf3_errors(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    shutil.copytree(FIXTURE, run_dir)
    iperf3 = json.loads((run_dir / "iperf3.json").read_text())
    iperf3["error"] = "unable to connect to server"
    (run_dir / "iperf3.json").write_text(json.dumps(iperf3) + "\n")
    result = run_checker(run_dir)
    report = json.loads((run_dir / "integrity.json").read_text())
    assert result.returncode == 1
    assert "iperf3 reported an error: unable to connect to server" in report["errors"]


@pytest.mark.parametrize("outcome", ["schema_invalid", "unactuatable"])
def test_checker_counts_nonactuated_decisions_without_control_fields(
    tmp_path: Path, outcome: str,
) -> None:
    run_dir = tmp_path / "run"
    shutil.copytree(FIXTURE, run_dir)
    rows = [json.loads(line) for line in (run_dir / "records.jsonl").read_text().splitlines()]
    row = rows[1]
    for field in (
        "policy_id", "control", "a1_http_status", "a1_put_ack_latency_s", "delta_a1_s",
        "a1_poll_wait_s", "delta_e2_s", "kpm_change_upper_bound_s", "kpm_change",
        "t_a1_put_requested", "t_a1_put_acknowledged", "t_seen", "t_control_sent",
        "t_gnb_acknowledged", "t_first_kpm_change", "remote_timestamps", "clock_probes",
    ):
        row[field] = None
    if outcome == "schema_invalid":
        row["decision_valid"] = False
        row["decision_error"] = "schema_validation"
        row["actuation_error"] = None
    else:
        row["decision_valid"] = True
        row["decision_error"] = None
        row["actuation_error"] = "deprioritise requires a stated priority"
    (run_dir / "records.jsonl").write_text("".join(json.dumps(value) + "\n" for value in rows))
    (run_dir / "records.a1.jsonl").write_text(
        (run_dir / "records.a1.jsonl").read_text().splitlines(keepends=True)[0]
    )
    (run_dir / "xapp.jsonl").write_text(
        "".join((run_dir / "xapp.jsonl").read_text().splitlines(keepends=True)[:2])
    )
    (run_dir / "gnb.log").write_text(
        "".join((run_dir / "gnb.log").read_text().splitlines(keepends=True)[:2])
    )

    result = run_checker(run_dir)
    report = json.loads((run_dir / "integrity.json").read_text())
    assert result.returncode == 0, result.stderr
    assert report["actual_decisions"] == 2
    assert report["actuated_decisions"] == 1
    assert report["invalid_decisions"] == (outcome == "schema_invalid")
    assert report["unactuatable_decisions"] == (outcome == "unactuatable")
    assert report["strict_gnb_exchanges"] == 1
    assert report["summary"]["delta_e2_s"]["null_count"] == 1


def test_checker_rejects_negative_interval_and_unpaired_gnb_request(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    shutil.copytree(FIXTURE, run_dir)
    rows = [json.loads(line) for line in (run_dir / "records.jsonl").read_text().splitlines()]
    rows[0]["delta_e2_s"] = -0.001
    (run_dir / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with (run_dir / "gnb.log").open("a") as handle:
        handle.write("2026-09-26T00:00:03.000000 [E2-DU   ] [I] Received RIC Control Request\n")
    result = run_checker(run_dir)
    report = json.loads((run_dir / "integrity.json").read_text())
    assert result.returncode == 1
    assert report["status"] == "FAIL"
    assert any("delta_e2_s is missing or negative" in error for error in report["errors"])
    assert "gNB log ends with an unacknowledged control request" in report["errors"]
