#!/usr/bin/env python3
"""Measure GET /health RTT through an already-established localhost SSH tunnel."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time
from typing import Any, Callable
from urllib.parse import urlparse
from urllib.request import urlopen


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def inclusive_p95(values: list[float]) -> float:
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[94]


def measure(
    url: str,
    count: int,
    *,
    opener: Callable[..., Any] = urlopen,
    clock: Callable[[], float] = time.perf_counter,
) -> list[float]:
    samples: list[float] = []
    for _ in range(count):
        started = clock()
        with opener(url, timeout=60.0) as response:
            body = response.read()
            if response.status != 200:
                raise RuntimeError(f"health GET returned HTTP {response.status}")
            if body.strip():
                json.loads(body)
        samples.append(clock() - started)
    return samples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="tunnel URL, e.g. http://127.0.0.1:18000/health")
    parser.add_argument("--phase", required=True, choices=("before", "after"))
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True, help="append-only JSONL output")
    args = parser.parse_args(argv)
    parsed = urlparse(args.url)
    if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost") or parsed.path != "/health":
        parser.error("--url must be an http://127.0.0.1:<port>/health or localhost tunnel URL")
    if args.count < 1:
        parser.error("--count must be positive")
    existing: list[dict[str, Any]] = []
    if args.output.exists():
        existing = [json.loads(line) for line in args.output.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(row.get("phase") == args.phase for row in existing):
        raise SystemExit(f"refusing duplicate {args.phase!r} phase in {args.output}")
    started = utc_now()
    samples_s = measure(args.url, args.count)
    record = {
        "phase": args.phase,
        "url": args.url,
        "count": args.count,
        "started_utc": started,
        "finished_utc": utc_now(),
        "median_ms": statistics.median(samples_s) * 1000.0,
        "p95_ms": inclusive_p95(samples_s) * 1000.0,
        "samples_ms": [value * 1000.0 for value in samples_s],
        "p95_method": "statistics.quantiles inclusive",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
    print(json.dumps({key: record[key] for key in ("phase", "count", "median_ms", "p95_ms")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
