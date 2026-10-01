#!/usr/bin/env python3
"""Launch the four hosted EXP-2026-003 RQ3 rate cells from the Mac.

OPENROUTER_API_KEY must already be exported. Each cell gets up to three separate CLI processes and ``__block<k>``
output directories, so degraded attempts remain available and the runner's non-resume refusal remains effective.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

HOSTED = ("Jev-1.13.0", "DeepSeek-V4.1-Flash", "GLM-5.3-Flash", "Qwen3.8-Flash")
MAX_BLOCKS = 3
SLUGS = {
    "Jev-1.13.0": "jev_1p13p0",
    "DeepSeek-V4.1-Flash": "deepseek_v4p1_flash",
    "GLM-5.3-Flash": "glm_5p3_flash",
    "Qwen3.8-Flash": "qwen3p8_flash",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP-2026-003 RQ3 hosted main-run launcher")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--client-location", required=True)
    parser.add_argument("--corpus-dir", default="data/ranbench/ranintent-v1")
    parser.add_argument("--interpreters", nargs="+", choices=HOSTED, default=list(HOSTED))
    parser.add_argument("--rates", nargs="+", type=float, default=[0.1, 0.5, 1.0, 2.0])
    parser.add_argument("--intents", type=int, default=300, help="300 main; smaller values are smoke-only")
    parser.add_argument("--spend-cap", type=float, default=20.0, help="USD per interpreter/rate cell")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)

    for model in args.interpreters:
        for rate in args.rates:
            rate_tag = f"{rate:g}".replace(".", "p")
            for block in range(1, MAX_BLOCKS + 1):
                out = args.out_root / SLUGS[model] / f"rate_{rate_tag}__block{block}"
                command = [
                    sys.executable,
                    "-m",
                    "src.ranbench.rq3_load",
                    "run",
                    "--corpus-dir",
                    args.corpus_dir,
                    "--interpreter",
                    model,
                    "--rate",
                    f"{rate:g}",
                    "--intents",
                    str(args.intents),
                    "--slots",
                    "4",
                    "--seed",
                    "20260925",
                    "--spend-cap",
                    str(args.spend_cap),
                    "--client-location",
                    args.client_location,
                    "--out",
                    str(out),
                ]
                if args.allow_dirty:
                    command.append("--allow-dirty")
                print(f"{model} rate={rate:g} block={block}/{MAX_BLOCKS} -> {out}", flush=True)
                completed = subprocess.run(command, check=False)
                if completed.returncode == 0:
                    print(f"COMPLETE {model} rate={rate:g} in block {block}", flush=True)
                    break
                try:
                    integrity = json.loads((out / "integrity.json").read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    integrity = {}
                if not integrity.get("rerun_block_required"):
                    print(f"FAILED {model} rate={rate:g} block={block} rc={completed.returncode}", flush=True)
                    return completed.returncode
                print(
                    f"DEGRADED {model} rate={rate:g} block={block}: {integrity.get('degradation_reason')} "
                    f"(retained at {out})",
                    flush=True,
                )
                if block == MAX_BLOCKS:
                    print(f"FAILED {model} rate={rate:g}: exhausted {MAX_BLOCKS} fresh blocks", flush=True)
                    return completed.returncode
                print(f"RETRY {model} rate={rate:g} as a new block", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
