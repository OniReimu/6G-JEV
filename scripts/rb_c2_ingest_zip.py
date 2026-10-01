#!/usr/bin/env python3
"""Receive C2 return ZIPs on the M4: verify, unpack, self-check, then move into place.

  python3 scripts/rb_c2_ingest_zip.py ZIP [ZIP ...] --dest c2-runs \
      --run-list FILE --matrix FILE --input-root DIR [--binary-sha256 HEX]

Per ZIP: the sha256 must match <zip>.sha256 and the member list must match
<zip>.manifest.txt (run_ids, file count, bytes). Members are extracted into a
hidden staging directory inside --dest, each run is self-checked with
rb_c2_verify_runs.py, and a PASS run is renamed atomically to <dest>/<run_id>.
An existing destination directory is never overwritten: a complete one is
reported SKIP_EXISTING, an incomplete one REFUSE_EXISTING. Exit 0 only if every
run is PASS or SKIP_EXISTING.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rb_c2_verify_runs as verify


def manifest_complete(run_dir: Path) -> bool:
    try:
        data = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return data.get("complete") is True and data.get("status") == "complete"


def read_zip_manifest(path: Path) -> dict[str, tuple[int, int]]:
    runs: dict[str, tuple[int, int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("run: "):
            parts = line.split()
            fields = dict(p.split("=", 1) for p in parts[2:])
            runs[parts[1]] = (int(fields["files"]), int(fields["bytes"]))
    return runs


def check_zip(zip_path: Path, allowed: set[str]) -> tuple[list[str], dict[str, tuple[int, int]]]:
    """Return (problems, per-run expected counts) before anything is extracted."""
    problems: list[str] = []
    sha_file = zip_path.with_name(zip_path.name + ".sha256")
    man_file = zip_path.with_name(zip_path.name + ".manifest.txt")
    for side in (sha_file, man_file):
        if not side.is_file():
            return [f"missing {side.name}"], {}
    want = sha_file.read_text(encoding="utf-8").split()[0]
    got = verify.sha256_file(zip_path)
    if want != got:
        return [f"sha256 mismatch: file {got}, .sha256 {want}"], {}
    expected = read_zip_manifest(man_file)
    seen: dict[str, list[int]] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            parts = PurePosixPath(info.filename).parts
            if (info.is_dir() or len(parts) != 2 or info.filename.startswith("/")
                    or ".." in parts or (info.external_attr >> 16) & 0o170000 == 0o120000):
                problems.append(f"unexpected member {info.filename!r}")
                continue
            count = seen.setdefault(parts[0], [0, 0])
            count[0] += 1
            count[1] += info.file_size
    for run_id in sorted(set(seen) | set(expected)):
        if run_id not in allowed:
            problems.append(f"{run_id} is not in the run list")
        if tuple(seen.get(run_id, (0, 0))) != expected.get(run_id):
            problems.append(f"{run_id}: zip has {seen.get(run_id)}, manifest.txt says {expected.get(run_id)}")
    return problems, expected


def ingest(zip_path: Path, dest: Path, run_ids: list[str], matrix: dict[str, dict[str, str]],
           input_root: Path, binary_sha256: str | None) -> list[tuple[str, str, list[str]]]:
    problems, expected = check_zip(zip_path, set(run_ids))
    if problems:
        return [(zip_path.name, "ZIP_REJECTED", problems)]
    dest.mkdir(parents=True, exist_ok=True)
    staging = dest / f".ingest-{zip_path.stem}"
    if staging.exists():
        note = f"staging {staging} exists from an earlier attempt; inspect and remove it first"
        return [(zip_path.name, "ZIP_REJECTED", [note])]
    staging.mkdir()
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(staging)
    results = []
    for run_id in sorted(expected):
        target = dest / run_id
        if target.exists():
            if manifest_complete(target):
                results.append((run_id, "SKIP_EXISTING", [f"{target} already complete; left untouched"]))
            else:
                results.append((run_id, "REFUSE_EXISTING", [f"{target} exists without a complete manifest"]))
            continue
        status, notes = verify.verify_run(staging / run_id, matrix[run_id], input_root, binary_sha256)
        if status == "PASS":
            (staging / run_id).rename(target)
        results.append((run_id, status, notes))
    # Remove only what was safely skipped; FAIL/REFUSE runs stay in staging for inspection.
    for run_id, status, _ in results:
        if status == "SKIP_EXISTING":
            shutil.rmtree(staging / run_id)
    if not any(staging.iterdir()):
        staging.rmdir()
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("zips", nargs="+", type=Path)
    parser.add_argument("--dest", required=True, type=Path)
    parser.add_argument("--run-list", required=True, type=Path)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--binary-sha256")
    args = parser.parse_args(argv)
    run_ids = verify.read_run_list(args.run_list)
    matrix = verify.read_matrix(args.matrix)
    results = []
    for zip_path in args.zips:
        print(f"== {zip_path.name}")
        batch = ingest(zip_path, args.dest, run_ids, matrix, args.input_root, args.binary_sha256)
        verify.print_table(batch)
        results += batch
    passed = all(status in ("PASS", "SKIP_EXISTING") for _, status, _ in results)
    print("RESULT " + ("PASS" if passed else "FAIL"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
