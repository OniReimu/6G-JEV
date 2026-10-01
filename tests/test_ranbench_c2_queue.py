from __future__ import annotations

import csv
import errno
import fcntl
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.ranbench.c2.queue import (
    check_binary,
    load_rows,
    manifest_complete,
    preflight,
    read_run_list,
    relocate_inputs,
    run_queue,
)


@pytest.fixture(autouse=True)
def ns3_build_tree(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "ns-3.48"
    (root / "cmake-cache").mkdir(parents=True)
    (root / "cmake-cache" / "CMakeCache.txt").write_text("", encoding="utf-8")
    monkeypatch.setenv("C2_NS3_ROOT", str(root))
    return root


def test_binary_pin_and_build_tree_fail_before_any_run(tmp_path: Path, monkeypatch) -> None:
    binary = tmp_path / "bin" / "ns3.48-ran-closed-loop-optimized"
    binary.parent.mkdir()
    binary.write_bytes(b"cd7590c6")
    check_binary(binary, hashlib.sha256(b"cd7590c6").hexdigest())
    with pytest.raises(ValueError, match="sha256"):
        check_binary(binary, "0" * 64)
    # A binary copied out of its build tree without C2_NS3_ROOT would crash run_c2.py
    # after the simulation; the queue refuses it up front.
    monkeypatch.delenv("C2_NS3_ROOT")
    with pytest.raises(FileNotFoundError, match="C2_NS3_ROOT"):
        check_binary(binary, None)


def row(tmp_path: Path, run_id: str, rng_run: str = "1") -> dict[str, str]:
    inputs = tmp_path / "inputs"
    inputs.mkdir(exist_ok=True)
    paths = {}
    for name, directory, suffix in (
        ("config_file", "configs", ".json"),
        ("schedule_file", "schedules", ".csv"),
        ("stream_file", "streams", ".json"),
    ):
        (inputs / directory).mkdir(exist_ok=True)
        path = inputs / directory / f"{run_id}-{name}{suffix}"
        path.write_text("{}\n" if suffix == ".json" else "t_s,scope,class,priority\n", encoding="utf-8")
        paths[name] = str(path)
    return {
        "run_id": run_id,
        "priority_tier": "1",
        "rng_run": rng_run,
        "simulated_seconds": "20",
        **paths,
    }


def write_matrix(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_load_rows_filters_and_rejects_unknown_ids(tmp_path: Path) -> None:
    rows = [row(tmp_path, "a"), row(tmp_path, "b")]
    rows[1]["priority_tier"] = "2"
    matrix = tmp_path / "matrix.csv"
    write_matrix(matrix, rows)
    assert [item["run_id"] for item in load_rows(matrix, {"b"}, {2})] == ["b"]
    try:
        load_rows(matrix, {"missing"})
    except ValueError as exc:
        assert "unknown run_id" in str(exc)
    else:
        raise AssertionError("unknown run ID accepted")


def test_run_list_order_is_preserved_and_duplicates_rejected(tmp_path: Path) -> None:
    rows = [row(tmp_path, "a"), row(tmp_path, "b")]
    matrix = tmp_path / "matrix.csv"
    write_matrix(matrix, rows)
    run_list = tmp_path / "runs.txt"
    run_list.write_text("# longest first\nb\na\n", encoding="utf-8")
    selected = read_run_list(run_list)
    assert selected == ["b", "a"]
    assert [item["run_id"] for item in load_rows(matrix, selected)] == ["b", "a"]
    run_list.write_text("a\na\n", encoding="utf-8")
    try:
        read_run_list(run_list)
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate run ID accepted")


def test_platform_input_root_relocation_preserves_filenames(tmp_path: Path) -> None:
    rows = [row(tmp_path, "a")]
    remote = Path("/srv/c2-production-inputs")
    relocate_inputs(rows, remote)
    assert rows[0]["config_file"] == str(remote / "configs" / "a-config_file.json")
    assert rows[0]["schedule_file"] == str(remote / "schedules" / "a-schedule_file.csv")
    assert rows[0]["stream_file"] == str(remote / "streams" / "a-stream_file.json")


def test_preflight_skips_only_verified_complete_and_requeues_invalid(
    tmp_path: Path, monkeypatch,
) -> None:
    rows = [row(tmp_path, name) for name in ("done", "partial", "ready", "missing")]
    output = tmp_path / "runs"
    (output / "done").mkdir(parents=True)
    (output / "done" / "manifest.json").write_text(
        json.dumps({"complete": True, "status": "complete"}), encoding="utf-8")
    (output / "partial").mkdir()
    (output / "partial" / "manifest.json").write_text(
        json.dumps({"complete": False, "status": "running"}), encoding="utf-8")
    Path(rows[-1]["stream_file"]).unlink()
    runner, binary = tmp_path / "runner.py", tmp_path / "binary"
    runner.write_text("", encoding="utf-8")
    binary.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        "src.ranbench.c2.queue.verify_run",
        lambda run_dir, *_args: ("PASS", []) if run_dir.name == "done" else ("FAIL", ["truncated"]),
    )

    results = preflight(rows, output, runner, binary)
    assert [(result.run_id, result.status) for result in results] == [
        ("done", "skip_complete"),
        ("partial", "ready"),
        ("ready", "ready"),
        ("missing", "missing_input"),
    ]
    assert manifest_complete(output / "done")
    assert not (output / "partial").exists()
    quarantined = list(output.glob("partial.incomplete-*"))
    assert len(quarantined) == 1 and not manifest_complete(quarantined[0])


def test_resume_quarantines_manifest_complete_run_that_fails_validation(
    tmp_path: Path, monkeypatch,
) -> None:
    rows = [row(tmp_path, "truncated")]
    output = tmp_path / "runs"
    run_dir = output / "truncated"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"complete": True, "status": "complete"}), encoding="utf-8")
    runner, binary = tmp_path / "runner.py", tmp_path / "binary"
    runner.write_text("", encoding="utf-8")
    binary.write_bytes(b"platform binary")
    expected_binary_sha = hashlib.sha256(binary.read_bytes()).hexdigest()

    def invalid(_run_dir, _row, input_root, binary_sha):
        assert input_root is None and binary_sha == expected_binary_sha
        return "FAIL", ["ue_slots.csv has 9 data rows, expected 10"]

    monkeypatch.setattr("src.ranbench.c2.queue.verify_run", invalid)
    result = preflight(rows, output, runner, binary)[0]
    assert result.status == "ready" and "ue_slots.csv" in result.detail
    assert not run_dir.exists()
    assert len(list(output.glob("truncated.incomplete-*"))) == 1


def test_dry_run_reports_quarantine_without_moving_directory(tmp_path: Path, monkeypatch) -> None:
    rows = [row(tmp_path, "partial")]
    output = tmp_path / "runs"
    run_dir = output / "partial"
    run_dir.mkdir(parents=True)
    marker = run_dir / "still-running"
    marker.write_text("keep", encoding="utf-8")
    runner, binary = tmp_path / "runner.py", tmp_path / "binary"
    runner.write_text("", encoding="utf-8")
    binary.write_text("", encoding="utf-8")
    monkeypatch.setattr("src.ranbench.c2.queue.verify_run", lambda *_args: ("INCOMPLETE", ["no manifest"]))

    result = run_queue(rows, output, runner, binary, jobs=1, dry_run=True)[0]
    assert result.status == "ready" and result.detail.startswith("would move INCOMPLETE run")
    assert marker.read_text(encoding="utf-8") == "keep"
    assert not list(output.glob("partial.incomplete-*"))
    assert not (output / ".locks").exists()


def test_held_run_lock_skips_without_touching_directory(tmp_path: Path, monkeypatch) -> None:
    rows = [row(tmp_path, "active")]
    output = tmp_path / "runs"
    run_dir = output / "active"
    run_dir.mkdir(parents=True)
    marker = run_dir / "still-running"
    marker.write_text("keep", encoding="utf-8")
    lock_path = output / ".locks" / "active.lock"
    lock_path.parent.mkdir()
    runner, binary = tmp_path / "runner.py", tmp_path / "binary"
    runner.write_text("", encoding="utf-8")
    binary.write_text("", encoding="utf-8")

    with lock_path.open("a+b") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = run_queue(rows, output, runner, binary, jobs=1)[0]
    assert result.status == "locked"
    assert marker.read_text(encoding="utf-8") == "keep"
    assert not list(output.glob("active.incomplete-*"))


def test_unsupported_flock_refuses_existing_invalid_run(tmp_path: Path, monkeypatch) -> None:
    rows = [row(tmp_path, "partial")]
    output = tmp_path / "runs"
    run_dir = output / "partial"
    run_dir.mkdir(parents=True)
    marker = run_dir / "still-running"
    marker.write_text("keep", encoding="utf-8")
    runner, binary = tmp_path / "runner.py", tmp_path / "binary"
    runner.write_text("", encoding="utf-8")
    binary.write_text("", encoding="utf-8")
    monkeypatch.setattr("src.ranbench.c2.queue.verify_run", lambda *_args: ("INCOMPLETE", ["no manifest"]))

    def unsupported(*_args) -> None:
        raise OSError(errno.ENOLCK, "flock unsupported")

    monkeypatch.setattr("src.ranbench.c2.queue.fcntl.flock", unsupported)
    result = run_queue(rows, output, runner, binary, jobs=1)[0]
    assert result.status == "refuse_existing"
    assert marker.read_text(encoding="utf-8") == "keep"
    assert not list(output.glob("partial.incomplete-*"))


def test_run_queue_parallel_completion_and_resume(tmp_path: Path, monkeypatch) -> None:
    rows = [row(tmp_path, "a"), row(tmp_path, "b")]
    output = tmp_path / "runs"
    runner, binary = tmp_path / "runner.py", tmp_path / "binary"
    runner.write_text("", encoding="utf-8")
    binary.write_text("", encoding="utf-8")

    def fake_run(cmd: list[str], **_kwargs) -> SimpleNamespace:
        run_dir = Path(cmd[cmd.index("--run-dir") + 1])
        run_dir.mkdir(parents=True)
        (run_dir / "manifest.json").write_text(
            json.dumps({"complete": True, "status": "complete"}), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("src.ranbench.c2.queue.subprocess.run", fake_run)
    monkeypatch.setattr("src.ranbench.c2.queue.verify_run", lambda *_args: ("PASS", []))
    first = run_queue(rows, output, runner, binary, jobs=2)
    assert [result.status for result in first] == ["complete", "complete"]
    second = run_queue(rows, output, runner, binary, jobs=2)
    assert [result.status for result in second] == ["skip_complete", "skip_complete"]


def test_run_queue_strictly_verifies_runner_output(tmp_path: Path, monkeypatch) -> None:
    rows = [row(tmp_path, "truncated")]
    output = tmp_path / "runs"
    runner, binary = tmp_path / "runner.py", tmp_path / "binary"
    runner.write_text("", encoding="utf-8")
    binary.write_bytes(b"platform binary")

    def fake_run(cmd: list[str], **_kwargs) -> SimpleNamespace:
        run_dir = Path(cmd[cmd.index("--run-dir") + 1])
        run_dir.mkdir(parents=True)
        (run_dir / "manifest.json").write_text(
            json.dumps({"complete": True, "status": "complete"}), encoding="utf-8")
        (run_dir / "ue_slots.csv").write_text(
            "t_slot,imsi,serving_cell,class,rx_bytes,n_rx_pkts,mean_delay_ms,p95_delay_ms\n",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("src.ranbench.c2.queue.subprocess.run", fake_run)
    result = run_queue(rows, output, runner, binary, jobs=1)[0]
    assert result.status == "failed"
    assert "ue_slots.csv has no data rows" in result.detail


def test_run_queue_fails_closed_when_any_preflight_row_is_bad(tmp_path: Path, monkeypatch) -> None:
    rows = [row(tmp_path, "ready"), row(tmp_path, "missing")]
    Path(rows[1]["stream_file"]).unlink()
    output = tmp_path / "runs"
    runner, binary = tmp_path / "runner.py", tmp_path / "binary"
    runner.write_text("", encoding="utf-8")
    binary.write_text("", encoding="utf-8")

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("queue launched work after a blocking preflight result")

    monkeypatch.setattr("src.ranbench.c2.queue.subprocess.run", must_not_run)
    results = run_queue(rows, output, runner, binary, jobs=2)
    assert [result.status for result in results] == ["ready", "missing_input"]
    assert not (output / "ready").exists()
