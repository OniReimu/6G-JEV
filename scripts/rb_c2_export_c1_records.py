#!/usr/bin/env python3
"""Export selected C1 rows as the per-interpreter records consumed by C2."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.ranbench.c1_analysis import (  # noqa: E402
    ACTUATED_FIELDS,
    EXPECTED_MODELS,
    PER_CASE_PRIMARY_LABEL,
    find_run_files,
    is_scored_correct,
    is_transport_ok,
    load_all_c1_runs,
)
from src.ranbench.c2.matrix import INTERPRETERS, slug  # noqa: E402

CONDITION = "c21_fresh"
FIXED_RUN_DIRS = ("c1_hosted", "c1_selfhosted", "c1_rerun")
QWEN_BLOCK_GLOBS = ("c1_hosted_qwen_block*", "c1_hosted_qwen_gapfill*")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_pool(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [str(row.get("intent_id")) for row in rows]
    if not rows:
        raise ValueError(f"empty pool: {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"pool has duplicate intent_id values: {path}")
    wrong = [row.get("intent_id") for row in rows if row.get("source_condition") != CONDITION]
    if wrong:
        raise ValueError(f"pool contains {len(wrong)} intent(s) outside {CONDITION}, e.g. {wrong[:3]}")
    return rows


def _analysis_run_dirs(runs_root: Path) -> list[Path]:
    dirs = [runs_root / name for name in FIXED_RUN_DIRS if (runs_root / name).is_dir()]
    for pattern in QWEN_BLOCK_GLOBS:
        dirs.extend(sorted(path for path in runs_root.glob(pattern) if path.is_dir()))
    if not dirs:
        raise FileNotFoundError(f"no C1 analysis run directories found under {runs_root}")
    return dirs


def _policy(row: dict[str, Any]) -> dict[str, Any] | None:
    if not bool(row.get("valid")) or not is_transport_ok(row):
        return None
    labels = row.get("labels")
    if isinstance(labels, list) and len(labels) == 1 and isinstance(labels[0], dict):
        return dict(labels[0])
    if isinstance(labels, dict):
        return dict(labels)
    raise ValueError(
        f"{row.get('model')} {row.get('case_id')}: valid selected row needs one policy label"
    )


def _export_row(row: dict[str, Any]) -> dict[str, Any]:
    valid = bool(row.get("valid")) and is_transport_ok(row)
    out: dict[str, Any] = {
        "model": row["model"],
        "case_id": str(row["case_id"]),
        "labels": _policy(row),
        "valid": valid,
        "latency_s": row.get("latency_s"),
    }
    if "queue_wait_s" in row:
        out["queue_wait_s"] = row["queue_wait_s"]
    return out


def _metrics(
    source_rows: list[dict[str, Any]],
    exported_rows: list[dict[str, Any]],
    pool_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    source_correct = sum(is_scored_correct(row, ACTUATED_FIELDS) for row in source_rows)
    exported_correct = sum(
        bool(row["valid"])
        and isinstance(row.get("labels"), dict)
        and all(row["labels"].get(field) == pool_by_id[row["case_id"]]["actuated"][field]
                for field in ACTUATED_FIELDS)
        for row in exported_rows
    )
    source_lats = [float(row["latency_s"]) for row in source_rows if row.get("latency_s") is not None]
    exported_lats = [float(row["latency_s"]) for row in exported_rows if row.get("latency_s") is not None]
    source = {
        "records": len(source_rows),
        "failures": sum(not bool(row.get("valid")) or not is_transport_ok(row) for row in source_rows),
        "actuated_vector_accuracy": source_correct / len(source_rows),
        "latency_median_s": float(np.median(source_lats)) if source_lats else None,
    }
    exported = {
        "records": len(exported_rows),
        "failures": sum(not row["valid"] for row in exported_rows),
        "actuated_vector_accuracy": exported_correct / len(exported_rows),
        "latency_median_s": float(np.median(exported_lats)) if exported_lats else None,
    }
    return {"source_selection": source, "exported": exported, "equal": source == exported}


def export_records(runs_root: Path, pool_path: Path, out_dir: Path) -> dict[str, Any]:
    """Select with the C1 analysis code, validate all models, then write the C2 records."""
    if tuple(EXPECTED_MODELS) != INTERPRETERS:
        raise ValueError(
            f"C1 model roster {tuple(EXPECTED_MODELS)!r} does not match C2 INTERPRETERS {INTERPRETERS!r}"
        )
    pool = _load_pool(pool_path)
    pool_by_id = {str(row["intent_id"]): row for row in pool}
    pool_ids = set(pool_by_id)
    discovered = find_run_files(_analysis_run_dirs(runs_root))
    loaded = load_all_c1_runs(
        discovered["ledgers"],
        block_of=discovered["block_of"],
        expected_case_ids={CONDITION: pool_ids},
    )

    files: dict[str, bytes] = {}
    per_model: dict[str, dict[str, Any]] = {}
    comparison: dict[str, dict[str, Any]] = {}
    for model in INTERPRETERS:
        selected = loaded.primary_by_cell.get((model, CONDITION), [])
        pool_selected = [row for row in selected if str(row["case_id"]) in pool_ids]
        selected_by_id = {str(row["case_id"]): row for row in pool_selected}
        if len(selected_by_id) != len(pool_selected):
            raise ValueError(f"{model}: duplicate selected rows for pool intents in {CONDITION}")
        missing = sorted(pool_ids - set(selected_by_id))
        if missing:
            raise ValueError(f"{model}: no selected C1 record for {len(missing)} pool intent(s), e.g. {missing[:3]}")
        ordered_source = [selected_by_id[str(item["intent_id"])] for item in pool]
        if any(row.get("model") != model for row in ordered_source):
            raise ValueError(f"{model}: selected C1 row model does not match C2 INTERPRETERS")
        exported = [_export_row(row) for row in ordered_source]
        result = _metrics(ordered_source, exported, pool_by_id)
        if not result["equal"]:
            raise ValueError(f"{model}: exported metrics differ from the selected C1 analysis rows")
        comparison[model] = result

        name = f"{slug(model)}.jsonl"
        data = "".join(json.dumps(row) + "\n" for row in exported).encode("utf-8")
        files[name] = data
        selected_blocks = sorted({str(row.get("_block_name")) for row in ordered_source})
        primary = loaded.primary_block_by_cell.get((model, CONDITION))
        rule = (
            "D-3: first transport-error-free attempt across C1 blocks in analysis order; "
            "first attempt is the failure row when none succeeds"
            if primary == PER_CASE_PRIMARY_LABEL
            else "C1 primary-block rule: original main run first, then complete rerun blocks by first send time"
        )
        per_model[model] = {
            "file": name,
            "sha256": hashlib.sha256(data).hexdigest(),
            "records": len(exported),
            "failures": result["exported"]["failures"],
            "selection_rule": rule,
            "primary_block": primary,
            "selected_blocks": selected_blocks,
        }

    manifest: dict[str, Any] = {
        "condition": CONDITION,
        "pool_case_mapping": "pool intent_id is copied from the c21_fresh corpus case_id",
        "pool": {"path": str(pool_path.resolve()), "sha256": _sha256(pool_path), "intents": len(pool)},
        "source_ledgers": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in discovered["ledgers"]
        ],
        "models": per_model,
        "comparison": comparison,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (out_dir / name).write_bytes(data)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export selected EXP-2026-003 C1 records for C2")
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = export_records(args.runs_root, args.pool, args.out_dir)
    print(f"{'model':<22} {'n':>4} {'fail':>5} {'C1 acc':>10} {'export acc':>10} {'C1 p50':>10} {'export p50':>10}")
    for model in INTERPRETERS:
        comp = manifest["comparison"][model]
        source, exported = comp["source_selection"], comp["exported"]
        print(
            f"{model:<22} {exported['records']:>4} {exported['failures']:>5} "
            f"{source['actuated_vector_accuracy']:>10.6f} {exported['actuated_vector_accuracy']:>10.6f} "
            f"{_fmt(source['latency_median_s']):>10} {_fmt(exported['latency_median_s']):>10}"
        )
    print(f"wrote {len(INTERPRETERS)} ledgers and manifest.json to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
