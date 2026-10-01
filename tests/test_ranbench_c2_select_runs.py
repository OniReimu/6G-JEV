from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest

from src.ranbench.c2.analysis import _verify, discover_run_dirs
from src.ranbench.c2.select_runs import main, select_runs
from test_ranbench_c2_return import SIM_S, UES, make_inputs, make_run

RUNS = ("r-orig", "r-rerun", "r-skip-bad", "r-missing")


@pytest.fixture
def env(tmp_path: Path) -> dict[str, Path]:
    inputs = tmp_path / "inputs"
    lines = ["run_id,ues,simulated_seconds,config_file,schedule_file,stream_file"]
    for run_id in RUNS:
        make_inputs(inputs, run_id)
        lines.append(f"{run_id},{UES},{SIM_S},{inputs}/configs/speed_3_ues_1.json,"
                     f"{inputs}/schedules/{run_id}.csv,{inputs}/streams/rate_0p1_speed_3.json")
    (tmp_path / "matrix.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    roots = {name: tmp_path / name for name in ("orig", "d7", "d7b")}
    return {"inputs": inputs, "matrix": tmp_path / "matrix.csv", "out": tmp_path / "sel", **roots}


def run(p: dict[str, Path], root: str, run_id: str, done: str | None, complete: bool = True) -> Path:
    run_dir = make_run(p[root], p["inputs"], run_id)
    manifest_path = run_dir / "manifest.json"
    if not complete:
        manifest_path.unlink()  # crashed or still running
        return run_dir
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["binary_sha256"] = root * 8
    if done:
        manifest["completed_at_utc"] = done
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir


def select(p: dict[str, Path], **kw) -> dict[str, dict[str, object]]:
    results = select_runs(p["matrix"], [p["orig"]], [p["d7"], p["d7b"]], p["out"], **kw)
    return {r["run_id"]: r for r in results}


def build(p: dict[str, Path]) -> None:
    # Original complete, but a rerun finished earlier: original wins.
    run(p, "orig", "r-orig", "2026-09-28T10:00:00+00:00")
    run(p, "d7", "r-orig", "2026-09-27T10:00:00+00:00")
    # Original crashed; d7b completed before d7: d7b wins.
    run(p, "orig", "r-rerun", None, complete=False)
    run(p, "d7", "r-rerun", "2026-09-29T12:00:00+00:00")
    run(p, "d7b", "r-rerun", "2026-09-29T20:00:00+11:00")  # 09:00 UTC: earliest, though later as a string
    # No original; the only other rerun copy is incomplete: the complete one wins.
    run(p, "d7", "r-skip-bad", None, complete=False)
    run(p, "d7b", "r-skip-bad", "2026-09-30T00:00:00+00:00")
    # r-missing: original and rerun both incomplete.
    run(p, "orig", "r-missing", None, complete=False)
    run(p, "d7", "r-missing", None, complete=False)


def test_selection_rules(env: dict[str, Path]) -> None:
    build(env)
    got = select(env)
    assert (got["r-orig"]["kind"], got["r-orig"]["chosen_root"]) == ("original", str(env["orig"]))
    assert (got["r-rerun"]["kind"], got["r-rerun"]["chosen_root"]) == ("rerun", str(env["d7b"]))
    assert got["r-rerun"]["binary_sha256"] == "d7b" * 8
    assert got["r-rerun"]["completed_at"] == "2026-09-29T20:00:00+11:00"
    assert got["r-skip-bad"]["chosen_root"] == str(env["d7b"])
    assert got["r-missing"]["kind"] == "" and got["r-missing"]["reason"].startswith("missing")
    assert "INCOMPLETE" in got["r-missing"]["other_copies"]

    out = env["out"]
    assert sorted(p.name for p in out.iterdir()) == ["r-orig", "r-rerun", "r-skip-bad", "selection.csv"]
    assert (out / "r-orig").resolve() == (env["orig"] / "r-orig").resolve()
    assert (out / "r-rerun").resolve() == (env["d7b"] / "r-rerun").resolve()
    assert (out / "r-rerun" / "manifest.json").is_file()
    with (out / "selection.csv").open(encoding="utf-8") as handle:
        rows = {r["run_id"]: r for r in csv.DictReader(handle)}
    assert rows["r-rerun"]["kind"] == "rerun" and rows["r-missing"]["chosen_root"] == ""


def test_incomplete_rerun_never_chosen(env: dict[str, Path]) -> None:
    run(env, "orig", "r-orig", None, complete=False)
    run(env, "d7", "r-orig", None, complete=False)
    assert select(env)["r-orig"]["kind"] == ""
    assert not (env["out"] / "r-orig").exists()


def test_complete_rerun_without_timestamp_is_refused(env: dict[str, Path]) -> None:
    run(env, "d7", "r-orig", None)
    run(env, "d7b", "r-orig", "2026-09-29T00:00:00+00:00")
    with pytest.raises(RuntimeError, match="without completed_at_utc"):
        select(env)


def test_out_dir_guard(env: dict[str, Path]) -> None:
    run(env, "orig", "r-orig", "2026-09-28T10:00:00+00:00")
    select(env)
    with pytest.raises(SystemExit, match="not empty"):
        select(env)
    assert select(env, force=True)["r-orig"]["kind"] == "original"
    (env["out"] / "stray").mkdir()
    with pytest.raises(SystemExit, match="refuses to remove"):
        select(env, force=True)


def test_analysis_accepts_selection_dir(env: dict[str, Path], capsys) -> None:
    build(env)
    main(["--matrix", str(env["matrix"]), "--original-root", str(env["orig"]),
          "--rerun-root", str(env["d7"]), "--rerun-root", str(env["d7b"]), "--out", str(env["out"])])
    assert "SUMMARY missing=1 original=1 rerun=2" in capsys.readouterr().out
    rows = [{"run_id": run_id} for run_id in RUNS]
    matrix_rows = list(csv.DictReader(env["matrix"].open(encoding="utf-8")))
    located = discover_run_dirs(rows, [env["out"]])
    assert set(located) == {"r-orig", "r-rerun", "r-skip-bad"}
    assert located["r-rerun"].run_dir == env["out"] / "r-rerun"
    status = _verify(matrix_rows, located, allow_incomplete=True)
    assert {k: v[0] for k, v in status.items()} == {
        "r-orig": "PASS", "r-rerun": "PASS", "r-skip-bad": "PASS", "r-missing": "MISSING"}


def test_restrict_excludes_other_roots_until_the_restricted_copy_completes(env: dict[str, Path], capsys) -> None:
    # D-16: old copies (an original and a rerun, one of them failing input verification) are EXCLUDED, never
    # verified, and the run is missing until the restricted root holds a complete copy.
    new = env["orig"].parent / "d16"
    old_orig = run(env, "orig", "r-orig", "2026-09-28T10:00:00+00:00")
    run(env, "d7", "r-orig", "2026-09-27T10:00:00+00:00")
    (old_orig / "schedule.csv").write_text("tampered\n", encoding="utf-8")
    run(env, "orig", "r-rerun", "2026-09-28T10:00:00+00:00")
    make_run(new, env["inputs"], "r-orig").joinpath("manifest.json").unlink()  # still running

    def pick(**kw):
        results = select_runs(env["matrix"], [env["orig"], new], [env["d7"], env["d7b"]], env["out"],
                              force=True, **kw)
        return {r["run_id"]: r for r in results}

    with pytest.raises(RuntimeError, match="more than one original root"):  # unrestricted: two originals
        pick()
    assert select(env)["r-orig"]["reason"].startswith("original FAIL")  # and the tampered one fails
    got = pick(restrict={"r-orig": new})
    assert got["r-orig"]["kind"] == "" and not (env["out"] / "r-orig").exists()
    assert got["r-orig"]["other_copies"] == (f"{new}:original:INCOMPLETE;{env['orig']}:original:EXCLUDED;"
                                             f"{env['d7']}:rerun:EXCLUDED")
    assert ":FAIL" not in got["r-orig"]["other_copies"]
    assert got["r-rerun"]["chosen_root"] == str(env["orig"])  # unrestricted ids are unchanged

    shutil.rmtree(new / "r-orig")  # the D-16 run completes
    run({**env, "d16": new}, "d16", "r-orig", "2026-10-01T00:00:00+00:00")
    main(["--matrix", str(env["matrix"]), "--original-root", str(env["orig"]), "--original-root", str(new),
          "--rerun-root", str(env["d7"]), "--out", str(env["out"]), "--force", "--restrict", f"r-orig={new}"])
    assert "SUMMARY missing=2 original=2" in capsys.readouterr().out
    assert (env["out"] / "r-orig").resolve() == (new / "r-orig").resolve()


def test_restrict_rejects_unknown_root_or_run(env: dict[str, Path]) -> None:
    with pytest.raises(SystemExit, match="not one of the runs roots"):
        select(env, restrict={"r-orig": env["orig"].parent / "elsewhere"})
    with pytest.raises(SystemExit, match="not a run id"):
        select(env, restrict={"r-nope": env["orig"]})
    with pytest.raises(SystemExit):
        main(["--matrix", str(env["matrix"]), "--original-root", str(env["orig"]), "--out", str(env["out"]),
              "--restrict", "r-orig"])
