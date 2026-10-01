from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.ranbench.ns3 import run_c2


def test_explicit_missing_schedule_exits_with_clear_error(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    missing = tmp_path / "missing.csv"
    run_dir = tmp_path / "run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_c2.py",
            "--config",
            str(config),
            "--run-dir",
            str(run_dir),
            "--schedule",
            str(missing),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        run_c2.main()

    assert exc_info.value.code == f"Error: schedule file not found: {missing}"
    assert not run_dir.exists()


def test_omitted_schedule_creates_empty_schedule(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    binary = tmp_path / "ran-closed-loop"
    binary.write_bytes(b"fake binary\n")
    run_dir = tmp_path / "run"

    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **_kwargs) -> SimpleNamespace:
        calls.append(cmd)
        output_arg = next(value for value in cmd if value.startswith("--outputDir="))
        output_dir = Path(output_arg.split("=", 1)[1])
        (output_dir / "streams.json").write_text(
            '{"rng_seed": 1, "blocks": {}, "ue_bounding_box": {}}\n',
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(run_c2, "build_facts", lambda _binary: {})
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_c2.py",
            "--config",
            str(config),
            "--run-dir",
            str(run_dir),
            "--binary",
            str(binary),
        ],
    )

    run_c2.main()

    schedule = run_dir / "schedule.csv"
    assert schedule.read_text(encoding="utf-8") == "t_s,scope,class,priority\n"
    assert calls == [
        [
            "/usr/bin/time",
            "-p",
            str(binary),
            f"--config={run_dir / 'config.json'}",
            f"--schedule={schedule}",
            f"--outputDir={run_dir}",
            "--rngRun=1",
        ]
    ]


def test_runner_refuses_existing_run_directory(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    binary = tmp_path / "ran-closed-loop"
    binary.write_bytes(b"fake binary\n")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    marker = run_dir / "user-data.txt"
    marker.write_text("preserve\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_c2.py",
            "--config",
            str(config),
            "--run-dir",
            str(run_dir),
            "--binary",
            str(binary),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        run_c2.main()

    assert exc_info.value.code == f"Error: refusing to overwrite existing run directory: {run_dir}"
    assert marker.read_text(encoding="utf-8") == "preserve\n"
