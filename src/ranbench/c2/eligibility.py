"""Production D-10 eligibility gate for one C2 design point."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from src.ranbench.c2 import stats
from src.ranbench.c2.loader import load_run
from src.ranbench.c2.pilot import PILOT_ARMS, analyse_controls, control_autocorr_times
from src.ranbench.c2.stats import resolved_positive
from src.ranbench.c2.verify import verify_run


def _matrix_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    run_ids = [row.get("run_id", "") for row in rows]
    if not rows or any(not run_id for run_id in run_ids):
        raise ValueError(f"matrix is empty or has a row without run_id: {path}")
    if len(run_ids) != len(set(run_ids)):
        raise ValueError(f"matrix has duplicate run_id values: {path}")
    return rows


def _control_rows(rows: list[dict[str, str]], design_point: str) -> dict[str, dict[str, str]]:
    point = [row for row in rows if row.get("design_point") == design_point]
    if not point:
        raise ValueError(f"design point not found in matrix: {design_point}")
    controls: dict[str, dict[str, str]] = {}
    for arm in PILOT_ARMS:
        matches = [row for row in point if row.get("arm") == arm and row.get("interpreter") == "control"]
        if not matches:
            raise ValueError(f"{design_point}: missing control {arm}")
        if len(matches) != 1:
            raise ValueError(f"{design_point}: found {len(matches)} control rows for {arm}")
        controls[arm] = matches[0]
    return controls


def _run_dir(run_id: str, roots: list[Path]) -> Path:
    matches = [root / run_id for root in roots if (root / run_id).is_dir()]
    if not matches:
        return roots[0] / run_id
    if len(matches) != 1:
        raise RuntimeError(f"{run_id}: run directory exists under multiple --runs-root values: {matches}")
    return matches[0]


def _events_path(row: dict[str, str]) -> Path:
    schedule = Path(row["schedule_file"])
    path = schedule.with_suffix(".events.json")
    if not path.is_file():
        raise RuntimeError(f"{row['run_id']}: control events missing: {path}")
    return path


def eligibility_flags(controls: dict[str, dict[str, Any]], block_pass: bool) -> dict[str, bool]:
    """Evaluate the D-9 floor and the original 5 pp sensitivity rule."""
    c1 = controls["C-1"]
    common = controls["C-2"]["pass"] and block_pass
    c1_d9 = c1["point"] is not None and c1["point"] > 0 and resolved_positive(c1, "p_raw")
    return {
        "eligible": bool(c1_d9 and common),
        "eligible_original_5pp": bool(c1["pass"] and common),
    }


def d10_eligibility_flags(
    controls: dict[str, dict[str, Any]], controls_10s: dict[str, dict[str, Any]]
) -> dict[str, bool]:
    """Report each block analysis and the D-10 both-block eligibility decisions."""
    primary = eligibility_flags(controls, block_pass=True)
    sensitivity = eligibility_flags(controls_10s, block_pass=True)
    return {
        **primary,
        "eligible_10s": sensitivity["eligible"],
        "eligible_d10": bool(primary["eligible"] and sensitivity["eligible"]),
        "eligible_original_5pp_10s": sensitivity["eligible_original_5pp"],
        "eligible_original_5pp_d10": bool(
            primary["eligible_original_5pp"] and sensitivity["eligible_original_5pp"]),
    }


def evaluate(matrix: Path, runs_roots: list[Path], design_point: str,
             block_s: float = 10.0) -> dict[str, Any]:
    """Strictly verify and evaluate the five controls for one production design point."""
    if not runs_roots:
        raise ValueError("at least one runs root is required")
    if block_s <= 0:
        raise ValueError("block length must be positive")
    rows = _control_rows(_matrix_rows(matrix), design_point)
    run_dirs = {arm: _run_dir(row["run_id"], runs_roots) for arm, row in rows.items()}

    manifests: dict[str, dict[str, Any]] = {}
    for arm, row in rows.items():
        status, problems = verify_run(run_dirs[arm], row, None)
        if status != "PASS":
            detail = "; ".join(problems)
            raise RuntimeError(f"{row['run_id']}: control run is not complete and valid ({status}): {detail}")
        manifests[arm] = json.loads((run_dirs[arm] / "manifest.json").read_text(encoding="utf-8"))

    binary_by_run = {rows[arm]["run_id"]: manifests[arm].get("binary_sha256") for arm in PILOT_ARMS}
    if any(not isinstance(value, str) or not value for value in binary_by_run.values()):
        raise RuntimeError("a control run manifest has no binary_sha256")
    if len(set(binary_by_run.values())) != 1:
        raise RuntimeError(f"{design_point}: control runs used different binary sha256 values")

    horizons = {float(manifest["sim_time_s"]) for manifest in manifests.values()}
    if len(horizons) != 1:
        raise RuntimeError(f"{design_point}: control runs have different simulated horizons")
    stream_shas = {manifest.get("stream_sha256") for manifest in manifests.values()}
    if len(stream_shas) != 1:
        raise RuntimeError(f"{design_point}: control runs used different streams")

    stream = json.loads((run_dirs["oracle"] / "stream.json").read_text(encoding="utf-8"))
    recorded_block = float(stream["meta"]["block_s"])
    events_by_arm = {
        arm: json.loads(_events_path(row).read_text(encoding="utf-8"))["events"]
        for arm, row in rows.items()
    }
    traces = {arm: load_run(run_dir) for arm, run_dir in run_dirs.items()}
    horizon = horizons.pop()
    autocorr = control_autocorr_times(traces, horizon)
    d10_assignments = stats.d10_block_assignments_from_stream(stream, horizon, autocorr)
    assignments = {
        "primary": d10_assignments["primary"],
        "sensitivity_10s": d10_assignments["sensitivity_10s"],
    }
    analysis = analyse_controls(
        stream, traces, events_by_arm, horizon, assignments, autocorr)
    controls = analysis["controls"]
    controls_10s = analysis["controls_by_block"]["sensitivity_10s"]
    eligibility = d10_eligibility_flags(controls, controls_10s)
    return {
        "design_point": design_point,
        "rule": "D-10",
        "run_ids": {arm: rows[arm]["run_id"] for arm in PILOT_ARMS},
        "binary_sha256_by_run": binary_by_run,
        "binary_sha256": next(iter(binary_by_run.values())),
        "tau_s": d10_assignments["tau_s"],
        "block_s": assignments["primary"]["block_s"],
        "n_blocks": assignments["primary"]["n_blocks"],
        "block_10s_s": assignments["sensitivity_10s"]["block_s"],
        "n_blocks_10s": assignments["sensitivity_10s"]["n_blocks"],
        **eligibility,
        "controls": controls,
        "controls_10s": controls_10s,
        "actuation": analysis["actuation"],
        "block_check": {
            "configured_block_s": assignments["primary"]["block_s"],
            "requested_block_s": block_s,
            "frozen_stream_block_s": recorded_block,
            "required_block_s": analysis["block_length_s"],
            "pass": True,
            "informational": "met by construction under D-10",
        },
        "autocorr_time_by_arm_s": analysis["autocorr_time_by_arm_s"],
        "block_length_s": analysis["block_length_s"],
        "block_length_by_arm_s": analysis["block_length_by_arm_s"],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate C2 D-10 eligibility at one production design point")
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--runs-root", action="append", required=True)
    parser.add_argument("--design-point", required=True)
    parser.add_argument(
        "--block", type=float, default=10.0,
        help="legacy informational value; D-10 computes B and always retains the registered 10 s sensitivity",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    result = evaluate(Path(args.matrix), [Path(root) for root in args.runs_root],
                      args.design_point, args.block)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
