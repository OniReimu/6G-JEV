"""Select one copy per C2 run_id across original and rerun roots (EXP-2026-003 D-7, D-11).

Rule (D-7 standby rule 2026-09-27 17:47, completion-order rule 2026-09-28 00:05, D-11): the
original run is used whenever it is complete; otherwise the complete rerun-root copy that
completed first. Complete = manifest.json exists and verify_run passes (the check analysis.py
applies). Completion time = manifest "completed_at_utc", written by run_c2.py. Outputs are never
read. The output directory holds one symlink per selected run and serves as a single --runs-root.

--restrict RUN_ID=ROOT (D-16): RUN_ID is taken only from ROOT, which must be a given root. Its copies under
the other roots are not verified and are listed as EXCLUDED; RUN_ID is missing until ROOT holds a complete copy.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path

from src.ranbench.c2.verify import read_matrix, verify_run

FIELDS = ["run_id", "chosen_root", "kind", "reason", "binary_sha256", "completed_at", "other_copies"]


def _copy(root: Path, run_id: str, row: dict[str, str], kind: str) -> dict[str, object]:
    run_dir = root / run_id
    status, _ = verify_run(run_dir, row, None)
    copy: dict[str, object] = {"root": root, "dir": run_dir, "kind": kind, "status": status,
                               "manifest": {}, "completed_at": None}
    if status == "PASS":
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        stamp = manifest.get("completed_at_utc")
        if stamp:
            when = datetime.fromisoformat(stamp)
            if when.tzinfo is None:
                raise ValueError(f"{run_dir}: completed_at_utc has no timezone: {stamp!r}")
            copy["completed_at"] = when
        copy["manifest"] = manifest
    return copy


def _label(copy: dict[str, object]) -> str:
    when = copy["completed_at"]
    return f"{copy['root']}:{copy['kind']}:{copy['status']}" + (f"@{when.isoformat()}" if when else "")


def select_one(run_id: str, row: dict[str, str], originals: list[Path], reruns: list[Path],
               only_root: Path | None = None) -> dict[str, object]:
    present = [(kind, root) for kind, roots in (("original", originals), ("rerun", reruns))
               for root in roots if (root / run_id).is_dir()]
    excluded = [f"{root}:{kind}:EXCLUDED" for kind, root in present
                if only_root is not None and root.absolute() != only_root.absolute()]
    copies = [_copy(root, run_id, row, kind) for kind, root in present
              if only_root is None or root.absolute() == only_root.absolute()]
    orig = [c for c in copies if c["kind"] == "original"]
    if len(orig) > 1:
        raise RuntimeError(f"{run_id}: present under more than one original root: {[str(c['root']) for c in orig]}")
    done_reruns = [c for c in copies if c["kind"] == "rerun" and c["status"] == "PASS"]
    if orig and orig[0]["status"] == "PASS":
        chosen, reason = orig[0], "original complete"
    elif done_reruns:
        undated = [str(c["dir"]) for c in done_reruns if c["completed_at"] is None]
        if undated:
            raise RuntimeError(f"{run_id}: complete rerun copy without completed_at_utc: {undated}")
        done_reruns.sort(key=lambda c: c["completed_at"])
        if len(done_reruns) > 1 and done_reruns[0]["completed_at"] == done_reruns[1]["completed_at"]:
            raise RuntimeError(f"{run_id}: two rerun copies share the earliest completed_at_utc")
        chosen = done_reruns[0]
        why = f"original {orig[0]['status']}" if orig else "original absent"
        reason = f"{why}; earliest of {len(done_reruns)} complete rerun copies"
    else:
        return {"run_id": run_id, "chosen": None, "chosen_root": "", "kind": "", "binary_sha256": "",
                "completed_at": "", "reason": "missing: no complete original or rerun copy",
                "other_copies": ";".join([_label(c) for c in copies] + excluded)}
    when = chosen["completed_at"]
    return {"run_id": run_id, "chosen": chosen["dir"], "chosen_root": str(chosen["root"]),
            "kind": chosen["kind"], "reason": reason,
            "binary_sha256": chosen["manifest"].get("binary_sha256", ""),
            "completed_at": when.isoformat() if when else "",
            "other_copies": ";".join([_label(c) for c in copies if c is not chosen] + excluded)}


def _prepare_out(out: Path, roots: list[Path], force: bool) -> None:
    if any(out.absolute() == root.absolute() for root in roots):
        raise SystemExit(f"--out {out} is one of the runs roots")
    out.mkdir(parents=True, exist_ok=True)
    entries = list(out.iterdir())
    if entries and not force:
        raise SystemExit(f"--out {out} is not empty (pass --force to replace an earlier selection)")
    for entry in entries:  # --force removes only what an earlier selection wrote
        if not (entry.is_symlink() or entry.name == "selection.csv"):
            raise SystemExit(f"--force refuses to remove {entry}: not a selection symlink")
        entry.unlink()


def select_runs(matrix: Path, originals: list[Path], reruns: list[Path], out: Path,
                force: bool = False, restrict: dict[str, Path] | None = None) -> list[dict[str, object]]:
    if set(map(Path.absolute, originals)) & set(map(Path.absolute, reruns)):
        raise SystemExit("a root is given both as --original-root and --rerun-root")
    restrict = restrict or {}
    rows = read_matrix(matrix)
    for run_id, root in restrict.items():
        if run_id not in rows:
            raise SystemExit(f"--restrict {run_id}: not a run id of the matrix")
        if root.absolute() not in set(map(Path.absolute, originals + reruns)):
            raise SystemExit(f"--restrict {run_id}={root}: not one of the runs roots")
    results = [select_one(run_id, row, originals, reruns, restrict.get(run_id)) for run_id, row in rows.items()]
    _prepare_out(out, originals + reruns, force)
    for result in results:
        if result["chosen"] is not None:
            os.symlink(os.path.abspath(result["chosen"]), out / result["run_id"], target_is_directory=True)
    with (out / "selection.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Select one copy per C2 run across original and rerun roots")
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--original-root", action="append", default=[])
    parser.add_argument("--rerun-root", action="append", default=[])
    parser.add_argument("--out", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--restrict", action="append", default=[], metavar="RUN_ID=ROOT",
                        help="take RUN_ID only from ROOT (one of the roots); other copies are EXCLUDED")
    args = parser.parse_args(argv)
    if not args.original_root:
        parser.error("at least one --original-root is required")
    restrict: dict[str, Path] = {}
    for item in args.restrict:
        run_id, sep, root = item.partition("=")
        if not sep or not run_id or not root or run_id in restrict:
            parser.error(f"--restrict {item!r}: expected RUN_ID=ROOT, once per run id")
        restrict[run_id] = Path(root)
    results = select_runs(Path(args.matrix), [Path(r) for r in args.original_root],
                          [Path(r) for r in args.rerun_root], Path(args.out), args.force, restrict)
    counts: dict[str, int] = {}
    for result in results:
        counts[result["kind"] or "missing"] = counts.get(result["kind"] or "missing", 0) + 1
    print("SUMMARY " + " ".join(f"{key}={value}" for key, value in sorted(counts.items())))
    for result in results:
        if result["kind"] != "original":
            print(f"{result['run_id']}: {result['kind'] or 'MISSING'}: {result['reason']} "
                  f"[{result['other_copies']}]")


if __name__ == "__main__":
    main()
