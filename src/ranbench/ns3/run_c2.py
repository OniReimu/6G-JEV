#!/usr/bin/env python3
"""
Thin runner for C2 5G-LENA closed-loop scenario (EXP-2026-003).
Stdlib only.
"""

import argparse
import hashlib
import json
import os
import re
import resource
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

NS3_ROOT = os.environ.get("NS3_ROOT", "ns-3.48")  # the ns-3.48 build tree (see src/ranbench/ns3/README.md)
DEFAULT_BINARY = NS3_ROOT + "/build/scratch/ns3.48-ran-closed-loop-optimized"
DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "c2-default.json")


def build_facts(binary: str) -> dict:
    """ns-3 version, LENA commit, compiler and build profile, read from the build tree."""
    inferred_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(binary))))
    ns3_root = os.environ.get("C2_NS3_ROOT", inferred_root)
    facts = {"ns3_version": os.path.basename(ns3_root)}
    facts["lena_commit"] = os.environ.get("C2_LENA_COMMIT", "")
    if not facts["lena_commit"]:
        try:
            facts["lena_commit"] = subprocess.run(
                ["git", "-C", ns3_root + "/contrib/nr", "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.SubprocessError) as e:
            facts["lena_commit"] = f"unknown: {e}"
    cache = {}
    with open(ns3_root + "/cmake-cache/CMakeCache.txt", encoding="utf-8") as f:
        for line in f:
            m = re.match(r"^([A-Za-z_]+):[A-Z]+=(.*)$", line.strip())
            if m:
                cache[m.group(1)] = m.group(2)
    cxx = cache.get("CMAKE_CXX_COMPILER", "c++")
    try:
        ver = subprocess.run([cxx, "--version"], capture_output=True, text=True,
                             check=True).stdout.splitlines()[0]
    except (IndexError, OSError, subprocess.SubprocessError) as e:
        ver = f"unknown: {e}"
    facts["compiler"] = f"{cxx}: {ver}"
    facts["build_profile"] = cache.get("build_profile", "unknown")
    facts["cmake_build_type"] = cache.get("CMAKE_BUILD_TYPE", "unknown")
    return facts


def sha256_file(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def time_command(binary: str) -> list[str]:
    """Portable external-time prefix: BSD on macOS, GNU on Linux."""
    time_bin = os.environ.get("C2_TIME_BIN") or shutil.which("time") or "/usr/bin/time"
    # BSD `time -l` calls a restricted sysctl after the child exits and can turn
    # a successful simulation into wrapper status 1. Peak RSS comes from
    # getrusage below, so portable POSIX output is sufficient on macOS.
    return [time_bin, "-p" if sys.platform == "darwin" else "-v", binary]


def main():
    parser = argparse.ArgumentParser(description="Run 5G-LENA C2 scenario")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to JSON config")
    parser.add_argument("--schedule", default=None, help="Path to CSV enforcement schedule")
    parser.add_argument("--stream", default=None, help="Optional frozen stream JSON to copy into the run")
    parser.add_argument("--run-dir", required=True, help="Directory to write run files and manifest")
    parser.add_argument("--binary", default=DEFAULT_BINARY, help="Path to compiled ns-3 binary")
    parser.add_argument("--rng-run", type=int, default=1, help="RNG run index")
    parser.add_argument("--sim-time", type=float, default=None, help="Optional simTime override in seconds")
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        sys.exit(f"Error: config file not found: {args.config}")
    # An explicit --schedule must exist; only an omitted --schedule means the empty schedule.
    if args.schedule is not None and not os.path.isfile(args.schedule):
        sys.exit(f"Error: schedule file not found: {args.schedule}")
    if args.stream is not None and not os.path.isfile(args.stream):
        sys.exit(f"Error: stream file not found: {args.stream}")
    if not os.path.isfile(args.binary):
        sys.exit(f"Error: binary not found: {args.binary}")

    run_dir = os.path.abspath(args.run_dir)
    if os.path.exists(run_dir):
        sys.exit(f"Error: refusing to overwrite existing run directory: {run_dir}")
    os.makedirs(run_dir, exist_ok=True)

    # Copy config
    target_config = os.path.join(run_dir, "config.json")
    shutil.copyfile(args.config, target_config)

    # Copy or create schedule
    target_schedule = os.path.join(run_dir, "schedule.csv")
    if args.schedule is not None:
        shutil.copyfile(args.schedule, target_schedule)
    else:
        with open(target_schedule, "w", encoding="utf-8") as f:
            f.write("t_s,scope,class,priority\n")

    target_stream = None
    stream_sha = None
    if args.stream is not None:
        target_stream = os.path.join(run_dir, "stream.json")
        shutil.copyfile(args.stream, target_stream)
        stream_sha = sha256_file(target_stream)
        sidecar = args.stream + ".sha256"
        if os.path.isfile(sidecar):
            shutil.copyfile(sidecar, target_stream + ".sha256")

    # Compute SHA-256 hashes
    binary_sha = sha256_file(args.binary)
    config_sha = sha256_file(target_config)
    schedule_sha = sha256_file(target_schedule)

    # Command line to run with the platform's external time utility.
    cmd = time_command(args.binary) + [
        f"--config={target_config}",
        f"--schedule={target_schedule}",
        f"--outputDir={run_dir}",
        f"--rngRun={args.rng_run}"
    ]
    if args.sim_time is not None:
        cmd.append(f"--simTime={args.sim_time}")

    print(f"[run_c2] Executing: {' '.join(cmd)}")
    t0 = time.perf_counter()
    # ns-3 writes hexagonal-topology.gnuplot into its working directory: keep it per run (the code
    # tree may be read-only, and concurrent runs would otherwise share one file).
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=run_dir)
    t1 = time.perf_counter()
    wall_time = t1 - t0

    # Write process logs
    stdout_path = os.path.join(run_dir, "stdout.log")
    stderr_path = os.path.join(run_dir, "stderr.log")
    with open(stdout_path, "w", encoding="utf-8") as f:
        f.write(proc.stdout)
    with open(stderr_path, "w", encoding="utf-8") as f:
        f.write(proc.stderr)

    if proc.returncode != 0:
        print(f"[run_c2] Error: simulation failed with return code {proc.returncode}")
        print("STDERR:")
        print(proc.stderr)
        sys.exit(proc.returncode)

    # Parse /usr/bin/time -l output from stderr
    time_wall = wall_time
    peak_rss = 0

    m_real = re.search(r"^\s*([0-9]+(?:\.[0-9]+)?)\s+real", proc.stderr, re.MULTILINE)
    if m_real:
        time_wall = float(m_real.group(1))

    m_rss = re.search(r"^\s*([0-9]+)\s+maximum resident set size", proc.stderr, re.MULTILINE)
    if m_rss:
        peak_rss = int(m_rss.group(1))
    else:
        # GNU time reports KiB; manifests always use bytes.
        m_rss = re.search(r"Maximum resident set size \(kbytes\):\s*([0-9]+)", proc.stderr)
        if m_rss:
            peak_rss = int(m_rss.group(1)) * 1024
    if not peak_rss:
        rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        peak_rss = int(rss if sys.platform == "darwin" else rss * 1024)

    with open(os.path.join(run_dir, "streams.json"), encoding="utf-8") as f:
        streams = json.load(f)

    manifest = dict(build_facts(args.binary))
    manifest.update({
        "binary_path": os.path.abspath(args.binary),
        "binary_sha256": binary_sha,
        "config_sha256": config_sha,
        "schedule_sha256": schedule_sha,
        "stream_sha256": stream_sha,
        "rng_run": args.rng_run,
        "sim_time_s": args.sim_time,
        "rng_seed": streams.get("rng_seed"),
        "stream_assignment": streams.get("blocks"),
        "ue_bounding_box": streams.get("ue_bounding_box"),
        "command": cmd,
        "wall_time_s": round(time_wall, 4),
        "peak_rss_bytes": peak_rss,
        "peak_rss_mb": round(peak_rss / (1024.0 * 1024.0), 2),
        "status": "complete",
        "complete": True,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    })

    manifest_path = os.path.join(run_dir, "manifest.json")
    manifest_tmp = manifest_path + ".tmp"
    with open(manifest_tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    os.replace(manifest_tmp, manifest_path)

    print(f"[run_c2] Completed successfully. Wall time: {time_wall:.2f}s, Peak RSS: {peak_rss / (1024*1024):.2f} MB")
    print(f"[run_c2] Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
