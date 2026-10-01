#!/usr/bin/env python3
"""Pack verified, not-yet-shipped C2 run directories into one return ZIP (stdlib only).

  python3 scripts/rb_c2_pack_zip.py --run-root DIR --run-list FILE --matrix FILE \
      --input-root DIR --outbox DIR [--binary-sha256 HEX] [--platform m5]

A batch is every run in the run list that passes rb_c2_verify_runs.py and is not
named in an existing <outbox>/*.manifest.txt. Writes, in the outbox:
  <platform>_EXP-2026-003_c2_bNN_<YYYYMMDD>.zip           run dirs as <run_id>/<file>
  <zip>.sha256                                            `shasum -a 256 -c` format
  <zip>.manifest.txt                                      run_ids, file count, bytes
Only regular files at the top of each run directory are added (no .DS_Store,
AppleDouble ._* files, extended attributes or resource forks).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rb_c2_verify_runs as verify

EXPERIMENT = "EXP-2026-003"


def run_files(run_dir: Path) -> list[Path]:
    return sorted(p for p in run_dir.iterdir()
                  if p.is_file() and not p.is_symlink()
                  and p.name != ".DS_Store" and not p.name.startswith("._"))


def shipped_run_ids(outbox: Path) -> tuple[set[str], int]:
    shipped: set[str] = set()
    last_batch = 0
    for path in outbox.glob("*.manifest.txt"):
        match = re.search(r"_b(\d+)_\d{8}\.zip\.manifest\.txt$", path.name)
        if match:
            last_batch = max(last_batch, int(match.group(1)))
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("run: "):
                shipped.add(line.split()[1])
    return shipped, last_batch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--run-list", required=True, type=Path)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--outbox", required=True, type=Path)
    parser.add_argument("--binary-sha256")
    parser.add_argument("--platform", default="m5")
    args = parser.parse_args(argv)

    args.outbox.mkdir(parents=True, exist_ok=True)
    shipped, last_batch = shipped_run_ids(args.outbox)
    results = verify.verify_many(args.run_root, verify.read_run_list(args.run_list),
                                 verify.read_matrix(args.matrix), args.input_root,
                                 args.binary_sha256)
    verify.print_table(results)
    failed = [r for r, s, _ in results if s == "FAIL"]
    if failed:
        print(f"REFUSED: {len(failed)} run(s) FAIL the self-check; fix or quarantine them first.")
        return 1
    batch = [r for r, s, _ in results if s == "PASS" and r not in shipped]
    if not batch:
        print("NOTHING TO PACK: every PASS run is already in an earlier ZIP.")
        return 0

    stamp = datetime.now().astimezone().strftime("%Y%m%d")  # local date of the packer
    name = f"{args.platform}_{EXPERIMENT}_c2_b{last_batch + 1:02d}_{stamp}.zip"
    zip_path = args.outbox / name
    tmp = args.outbox / (name + ".part")
    per_run: list[tuple[str, int, int]] = []
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for run_id in batch:
            files = run_files(args.run_root / run_id)
            for path in files:
                zf.write(path, f"{run_id}/{path.name}")
            per_run.append((run_id, len(files), sum(p.stat().st_size for p in files)))
    tmp.replace(zip_path)
    digest = verify.sha256_file(zip_path)
    (args.outbox / (name + ".sha256")).write_text(f"{digest}  {name}\n", encoding="utf-8")
    shas = sorted({str(json.loads((args.run_root / r / "manifest.json").read_text(encoding="utf-8"))
                       .get("binary_sha256")) for r in batch})
    lines = [
        f"zip: {name}",
        f"zip_sha256: {digest}",
        f"created_utc: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"binary_sha256: {','.join(shas)}",
        f"runs: {len(per_run)}",
        f"files: {sum(n for _, n, _ in per_run)}",
        f"uncompressed_bytes: {sum(b for _, _, b in per_run)}",
    ] + [f"run: {r} files={n} bytes={b}" for r, n, b in per_run]
    (args.outbox / (name + ".manifest.txt")).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"PACKED {len(per_run)} run(s) -> {zip_path} ({zip_path.stat().st_size} bytes)")
    print(f"sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
