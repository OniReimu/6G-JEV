"""D-10 heap-layout robustness check for the C2 base point."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.ranbench.c2 import stats
from src.ranbench.c2.analysis import _events, _matrix_rows, _one
from src.ranbench.c2.loader import load_run
from src.ranbench.c2.outcomes import W_PRIMARY, SlotTable, actuation, intent_outcomes
from src.ranbench.c2.verify import verify_run

HOSTED_MODELS = ("DeepSeek-V4.1-Flash", "GLM-5.3-Flash", "Qwen3.8-Flash")
ANCHOR = "Jev-1.13.0"


def layout_flag(estimates: list[float]) -> dict[str, Any]:
    """Apply D-10's sign-disagreement and production-relative range rules."""
    if not estimates or any(not np.isfinite(value) for value in estimates):
        raise ValueError("layout estimates must be a non-empty list of finite values")
    signs = {int(np.sign(value)) for value in estimates}
    estimate_range = float(max(estimates) - min(estimates))
    sign_differs = len(signs) > 1
    range_exceeds_production = estimate_range > abs(estimates[0])
    return {
        "range_min": float(min(estimates)),
        "range_max": float(max(estimates)),
        "range": estimate_range,
        "sign_differs": sign_differs,
        "range_exceeds_production": range_exceeds_production,
        "not_robust": bool(sign_differs or range_exceeds_production),
    }


def _selected_rows(matrix: Path, design_point: str) -> dict[str, dict[str, str]]:
    point_rows = [row for row in _matrix_rows(matrix) if row.get("design_point") == design_point]
    if not point_rows:
        raise ValueError(f"design point not found in matrix: {design_point}")
    selected = {
        "oracle": _one(point_rows, arm="oracle", interpreter="control"),
        "no-update": _one(point_rows, arm="no-update", interpreter="control"),
        ANCHOR: _one(point_rows, arm="L", mode="E", interpreter=ANCHOR),
    }
    selected.update({
        model: _one(point_rows, arm="L", mode="E", interpreter=model)
        for model in HOSTED_MODELS
    })
    return selected


def common_binary_sha256(
    roots: list[tuple[str, Path]], rows: dict[str, dict[str, str]]
) -> str:
    """The one binary_sha256 shared by every selected arm in every root; raise otherwise.

    Read from each run's manifest.json. A missing manifest or hash, or any difference across arms
    or roots, is an error; there is no incomplete-run exemption."""
    by_run: dict[str, str] = {}
    for name, root in roots:
        for arm, row in rows.items():
            label = f"{name}:{arm} ({root / row['run_id']})"
            manifest_path = root / row["run_id"] / "manifest.json"
            if not manifest_path.is_file():
                raise RuntimeError(f"{label}: manifest.json missing, binary_sha256 unavailable")
            value = json.loads(manifest_path.read_text(encoding="utf-8")).get("binary_sha256")
            if not isinstance(value, str) or not value:
                raise RuntimeError(f"{label}: manifest has no binary_sha256")
            by_run[label] = value
    if len(set(by_run.values())) != 1:
        detail = "; ".join(f"{label}={value}" for label, value in by_run.items())
        raise RuntimeError(f"layout check requires one binary_sha256 across all arms and roots: {detail}")
    return next(iter(by_run.values()))


def _load_selected(
    root: Path, rows: dict[str, dict[str, str]], allow_incomplete: bool
) -> tuple[dict[str, Any], dict[str, str]]:
    traces: dict[str, Any] = {}
    statuses: dict[str, str] = {}
    for arm, row in rows.items():
        run_dir = root / row["run_id"]
        status, problems = verify_run(run_dir, row, None)
        statuses[arm] = status
        if status != "PASS" and not allow_incomplete:
            raise RuntimeError(
                f"{root.name}/{row['run_id']}: run is not complete and valid "
                f"({status}): {'; '.join(problems)}")
        if not run_dir.is_dir():
            raise RuntimeError(f"{root.name}/{row['run_id']}: run directory is missing")
        traces[arm] = load_run(run_dir, tolerate_torn_tail=allow_incomplete)
    return traces, statuses


def _affected_arrays(
    rows: dict[str, dict[str, str]], traces: dict[str, Any], horizon: float
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for arm, row in rows.items():
        events = _events(row)
        outcomes = intent_outcomes(
            traces[arm], events, SlotTable(traces[arm]), t_end=horizon)
        for outcome in outcomes:
            if outcome["t_issue_s"] + W_PRIMARY > horizon + 1e-9:
                outcome["aff_W2"] = None
        arrays[arm] = stats.as_array(outcomes, "aff_W2")
    return arrays


def _point_estimates(
    rows: dict[str, dict[str, str]], traces: dict[str, Any]
) -> dict[str, float]:
    horizon = min(
        min(trace.sim_time_s, float(trace.ue_slots["t"].max()))
        for trace in traces.values() if len(trace.ue_slots)
    )
    arrays = _affected_arrays(rows, traces, horizon)
    oracle_events = _events(rows["oracle"])
    directions = np.array([
        row["direction"] for row in actuation(
            oracle_events, traces["oracle"].clusters,
            traces["oracle"].config["priorities"])
    ], dtype=float)
    actuating = directions != 0
    no_update = np.where(actuating, directions * arrays["no-update"], np.nan)
    oracle = np.where(actuating, directions * arrays["oracle"], np.nan)
    common = np.isfinite(no_update) & np.isfinite(oracle)
    if not common.any():
        raise RuntimeError("C-1 has no defined paired actuating-intent outcomes")
    estimates = {"C-1": float((no_update[common] - oracle[common]).mean())}
    for model in HOSTED_MODELS:
        common = np.isfinite(arrays[model]) & np.isfinite(arrays[ANCHOR])
        if not common.any():
            raise RuntimeError(f"{model} vs {ANCHOR} has no defined paired outcomes")
        estimates[f"{model} vs {ANCHOR}"] = float(
            (arrays[model][common] - arrays[ANCHOR][common]).mean())
    return estimates


def analyse(
    matrix: Path, design_point: str, production_root: Path, layout_roots: list[Path],
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    if len(layout_roots) != 3:
        raise ValueError("D-10 requires exactly three layout roots")
    rows = _selected_rows(matrix, design_point)
    roots = [("production", production_root)] + [
        (f"layout_{index}", root) for index, root in enumerate(layout_roots, start=1)
    ]
    binary_sha256 = common_binary_sha256(roots, rows)
    realizations = []
    for name, root in roots:
        traces, statuses = _load_selected(root, rows, allow_incomplete)
        realizations.append({
            "name": name,
            "root": str(root),
            "run_status_by_arm": statuses,
            "estimates": _point_estimates(rows, traces),
        })
    contrast_names = ["C-1"] + [f"{model} vs {ANCHOR}" for model in HOSTED_MODELS]
    contrasts = []
    for contrast in contrast_names:
        estimates = [realization["estimates"][contrast] for realization in realizations]
        contrasts.append({
            "contrast": contrast,
            "production_estimate": estimates[0],
            "estimates_by_realization": {
                realization["name"]: estimate
                for realization, estimate in zip(realizations, estimates)
            },
            **layout_flag(estimates),
        })
    return {
        "rule": "D-10",
        "design_point": design_point,
        "selected_run_ids": {arm: row["run_id"] for arm, row in rows.items()},
        "binary_sha256": binary_sha256,
        "realizations": realizations,
        "contrasts": contrasts,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the C2 D-10 base-point layout robustness check")
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--design-point", required=True)
    parser.add_argument("--production-root", required=True)
    parser.add_argument("--layout-root", action="append", required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    result = analyse(
        Path(args.matrix), args.design_point, Path(args.production_root),
        [Path(root) for root in args.layout_root], args.allow_incomplete)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
