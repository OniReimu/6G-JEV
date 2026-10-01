"""Unix-socket RQ5 live/replay controller and ns-3 process launcher."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import resource
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.ranbench.c2.queue import (
    check_binary,
    load_rows,
    manifest_complete,
    relocate_inputs,
)
from src.ranbench.interpreters.manifest import build_interpreter
from src.ranbench.ns3.run_c2 import build_facts
from src.ranbench.rq5.interpreters import FakeRq5Interpreter, query_interpreter
from src.ranbench.rq5.protocol import (
    RQ5_DEADLINE_S,
    Rq5Result,
    canonical_snapshot,
    normalise_actions,
)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")


def _load_replay(path: Path) -> tuple[dict[float, dict[str, Any]], dict[float, str]]:
    decisions: dict[float, dict[str, Any]] = {}
    for line in (path / "decisions.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if isinstance(row.get("actions"), list):
                decisions[float(row["tick_s"])] = row
    snapshots = {
        float(row["tick_s"]): canonical_snapshot(row)
        for row in (
            json.loads(line) for line in (path / "snapshots.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    return decisions, snapshots


def serve_connection(
    conn: socket.socket,
    out_dir: Path,
    interpreter: Any | None = None,
    replay_dir: Path | None = None,
) -> None:
    """Serve one simulator connection. Live and replay emit the identical response contract."""
    replay_decisions: dict[float, dict[str, Any]] = {}
    replay_snapshots: dict[float, str] = {}
    if replay_dir is not None:
        replay_decisions, replay_snapshots = _load_replay(replay_dir)
    with conn.makefile("r", encoding="utf-8", newline="\n") as reader, conn.makefile(
        "w", encoding="utf-8", newline="\n"
    ) as writer:
        for line in reader:
            message = json.loads(line)
            kind = message.get("type")
            tick = float(message["tick_s"])
            if kind == "skipped":
                _append_jsonl(out_dir / "decisions.jsonl", {**message, "status": "skipped"})
                writer.write(json.dumps({"type": "ack", "tick_s": tick}, separators=(",", ":")) + "\n")
                writer.flush()
                continue
            if kind != "snapshot":
                raise RuntimeError(f"unknown simulator message type {kind!r}")
            snapshot = {key: value for key, value in message.items() if key != "type"}
            _append_jsonl(out_dir / "snapshots.jsonl", snapshot)
            if replay_dir is not None:
                if tick not in replay_decisions:
                    raise RuntimeError(f"replay has no decision for tick {tick:.6f}")
                if replay_snapshots.get(tick) != canonical_snapshot(snapshot):
                    raise RuntimeError(f"snapshot mismatch at replay tick {tick:.6f}")
                recorded = replay_decisions[tick]
                result = Rq5Result(
                    actions=recorded["actions"], invalid_fields=int(recorded["invalid_fields"]),
                    error_type=recorded.get("error_type"), metadata=recorded.get("metadata", {}),
                )
                latency_s = float(recorded["latency_s"])
                timed_out = recorded.get("timed_out") is True
                mode = "replay"
            else:
                if interpreter is None:
                    raise RuntimeError("live controller needs an interpreter")
                start = time.perf_counter()
                result = query_interpreter(interpreter, snapshot)
                wall_latency_s = time.perf_counter() - start
                adapter_latency = result.metadata.get("adapter_latency_s")
                post_timeout_wait = result.metadata.get("post_timeout_wait_s")
                # A stale keep-alive retry is timed from the retry's send: drop the failed attempt.
                failed_attempt = result.metadata.get("failed_attempt_s")
                untimed = result.metadata.get("untimed_s")  # adapter bookkeeping (AnyJev setup, token count)
                latency_s = max(0.0, wall_latency_s - float(post_timeout_wait or 0.0) - float(failed_attempt or 0.0)
                                - float(untimed or 0.0))
                if isinstance(adapter_latency, (int, float)):
                    latency_s = max(latency_s, float(adapter_latency))
                timed_out = result.error_type == "TimeoutError" or latency_s >= RQ5_DEADLINE_S
                if timed_out:
                    held = normalise_actions(snapshot, [], "TimeoutError")
                    result = Rq5Result(
                        held.actions, held.invalid_fields, "TimeoutError",
                        {**result.metadata, "timed_out": True},
                    )
                    latency_s = RQ5_DEADLINE_S
                mode = "live"
                _append_jsonl(out_dir / "calls.jsonl", {
                    "tick_s": tick, "family": "rq5_percell", "interpreter": interpreter.name,
                    "latency_s": latency_s, "wall_latency_s": wall_latency_s,
                    "deadline_s": RQ5_DEADLINE_S, "timed_out": timed_out,
                    "error_type": result.error_type, **result.metadata,
                })
            decision = {
                "tick_s": tick, "status": "skipped" if timed_out else "decision", "mode": mode,
                "latency_s": latency_s, "timed_out": timed_out,
                "invalid_fields": result.invalid_fields, "error_type": result.error_type,
                "actions": result.actions, "metadata": result.metadata,
            }
            _append_jsonl(out_dir / "decisions.jsonl", decision)
            response = {key: decision[key] for key in ("tick_s", "latency_s", "invalid_fields", "actions")}
            response["type"] = "decision"
            writer.write(json.dumps(response, separators=(",", ":"), allow_nan=False) + "\n")
            writer.flush()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_metrics(decisions: list[dict[str, Any]]) -> dict[str, int | float]:
    return {
        "rq5_ticks": len(decisions),
        "rq5_decisions": sum(row.get("status") == "decision" for row in decisions),
        "rq5_skipped": sum(row.get("status") == "skipped" for row in decisions),
        "rq5_timeouts": sum(row.get("timed_out") is True for row in decisions),
        "rq5_deadline_s": RQ5_DEADLINE_S,
        "rq5_invalid_fields": sum(int(row.get("invalid_fields", 0)) for row in decisions),
        "rq5_retried_calls": sum((row.get("metadata") or {}).get("retried") is True for row in decisions),
        "rq5_error_ticks": sum(row.get("error_type") is not None for row in decisions
                               if isinstance(row.get("actions"), list)),
    }


def run_simulator(
    *, binary: Path, config: Path, schedule: Path, run_dir: Path, rng_run: int,
    sim_time: float | None, interpreter: Any | None, replay_dir: Path | None,
    stream: Path | None = None, wrapper: list[str] | None = None, run_id: str | None = None,
) -> dict[str, Any]:
    """One arm-(b) run: copy inputs, serve the hook socket, write manifest.json (complete) last."""
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    config_copy, schedule_copy = run_dir / "config.json", run_dir / "schedule.csv"
    shutil.copy2(config, config_copy)
    shutil.copy2(schedule, schedule_copy)
    stream_copy = None
    if stream is not None:
        stream_copy = run_dir / "stream.json"
        shutil.copy2(stream, stream_copy)
        sidecar = Path(str(stream) + ".sha256")
        if sidecar.is_file():
            shutil.copy2(sidecar, Path(str(stream_copy) + ".sha256"))
    # Short path: sun_path is 104 (macOS) / 108 (Linux) bytes.
    socket_path = Path(tempfile.gettempdir()) / f"rq5-{os.getpid()}-{uuid.uuid4().hex[:8]}.sock"
    socket_path.unlink(missing_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(socket_path))
    server.listen(1)
    server_error: list[BaseException] = []

    def _server() -> None:
        try:
            conn, _ = server.accept()
            with conn:
                serve_connection(conn, run_dir, interpreter=interpreter, replay_dir=replay_dir)
        except Exception as exc:  # noqa: BLE001 - transfer the server-thread failure to the caller
            server_error.append(exc)

    thread = threading.Thread(target=_server, name="rq5-controller", daemon=True)
    thread.start()
    command = list(wrapper or []) + [
        str(binary), f"--config={config_copy}", f"--schedule={schedule_copy}",
        f"--outputDir={run_dir}", f"--rngRun={rng_run}", f"--rq5Socket={socket_path}",
    ]
    if sim_time is not None:
        command.append(f"--simTime={sim_time}")
    started = time.perf_counter()
    try:
        with (run_dir / "stdout.log").open("w", encoding="utf-8") as out, \
                (run_dir / "stderr.log").open("w", encoding="utf-8") as err:
            # cwd = run_dir: ns-3 writes hexagonal-topology.gnuplot to its working directory.
            returncode = subprocess.run(command, stdout=out, stderr=err, cwd=run_dir, check=False).returncode
        wall = time.perf_counter() - started
        thread.join(timeout=5)
    finally:
        server.close()
        socket_path.unlink(missing_ok=True)
    if returncode != 0:
        # A controller failure closes the socket and makes ns-3 abort; report the controller's error, which is
        # the cause, rather than only the exit code.
        cause = f"RQ5 controller failed: {server_error[0]!r}; " if server_error else ""
        raise RuntimeError(f"{cause}ns-3 exited {returncode}; see {run_dir / 'stderr.log'}")
    if thread.is_alive():
        raise RuntimeError("RQ5 controller did not finish after ns-3 exited")
    if server_error:
        raise RuntimeError(f"RQ5 controller failed: {server_error[0]}")
    rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    peak_rss = int(rss if sys.platform == "darwin" else rss * 1024)
    streams = json.loads((run_dir / "streams.json").read_text(encoding="utf-8"))
    decisions = [
        json.loads(line) for line in (run_dir / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ] if (run_dir / "decisions.jsonl").is_file() else []
    manifest = dict(build_facts(str(binary)))
    manifest.update({
        "run_id": run_id, "rq5_arm": "b-requery-1s",
        "mode": "replay" if replay_dir else "live",
        "interpreter": getattr(interpreter, "name", None),
        "binary_path": str(binary.resolve()), "binary_sha256": _sha256(binary),
        "config_sha256": _sha256(config_copy), "schedule_sha256": _sha256(schedule_copy),
        "stream_sha256": _sha256(stream_copy) if stream_copy else None,
        "rng_run": rng_run, "sim_time_s": sim_time, "rng_seed": streams.get("rng_seed"),
        "stream_assignment": streams.get("blocks"), "ue_bounding_box": streams.get("ue_bounding_box"),
        "command": command, "wall_time_s": round(wall, 4), "peak_rss_bytes": peak_rss,
        "peak_rss_mb": round(peak_rss / (1024.0 * 1024.0), 2),
        **_manifest_metrics(decisions),
        "replay_source": str(replay_dir) if replay_dir else None,
        "status": "complete", "complete": True,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    tmp = run_dir / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, run_dir / "manifest.json")
    return manifest


def resolve_row_inputs(row: dict[str, str], by_id: dict[str, dict[str, str]]) -> dict[str, Any]:
    """An RQ5 (b) row runs on its reused arm-(a) row's config, stream and N-arm schedule (same installs)."""
    if row.get("rqs") != "RQ5":
        raise ValueError(f"{row['run_id']}: not an RQ5 row")
    ref = row.get("reused_arm_a_run_id", "")
    if ref not in by_id:
        raise ValueError(f"{row['run_id']}: reused arm-(a) row {ref!r} is not in the matrix")
    a_row = by_id[ref]
    if a_row["interpreter"] != row["interpreter"] or a_row["arm"] != "N" or a_row["design_point"] != row["design_point"]:
        raise ValueError(f"{row['run_id']}: arm-(a) row {ref} is not this interpreter's base-point N arm")
    if abs(float(a_row["simulated_seconds"]) - float(row["simulated_seconds"])) > 1e-9:
        raise ValueError(f"{row['run_id']}: simulated_seconds differs from arm (a)")
    return {
        "config": Path(a_row["config_file"]), "schedule": Path(a_row["schedule_file"]),
        "stream": Path(a_row["stream_file"]), "sim_time": float(row["simulated_seconds"]),
        "rng_run": int(row["rng_run"]), "interpreter": row["interpreter"],
    }


def make_interpreter(name: str, base_url: str | None, manifest_path: Path | None) -> Any:
    if name.startswith("fake-rq5"):
        _, _, seed = name.partition(":")
        return FakeRq5Interpreter(seed=int(seed or 0))
    interpreter = build_interpreter(
        name, api_key=os.environ.get("OPENROUTER_API_KEY"), base_url=base_url,
        manifest=None if manifest_path is None else json.loads(manifest_path.read_text(encoding="utf-8")),
    )
    interpreter.timeout_s = RQ5_DEADLINE_S
    return interpreter


def _archive_incomplete(path: Path) -> Path:
    target = Path(str(path) + ".incomplete")
    if target.exists():
        target = Path(str(target) + f"-{uuid.uuid4().hex[:8]}")
    os.replace(path, target)
    return target


def prepare_attempt(run_dir: Path) -> Path | None:
    """Archive partial work and return the next private attempt, or None for a complete run."""
    if manifest_complete(run_dir):
        return None
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(run_dir.name)}\.attempt-(\d+)(?:\..*)?$")
    siblings = list(run_dir.parent.glob(f"{run_dir.name}.attempt-*"))
    numbers = [int(match.group(1)) for path in siblings if (match := pattern.match(path.name))]
    next_number = max(numbers, default=0) + 1

    if run_dir.exists():
        target = Path(f"{run_dir}.attempt-{next_number}.incomplete")
        if target.exists():
            target = Path(str(target) + f"-{uuid.uuid4().hex[:8]}")
        os.replace(run_dir, target)
        next_number += 1

    active = [path for path in siblings if re.fullmatch(rf"{re.escape(run_dir.name)}\.attempt-\d+", path.name)]
    complete = [path for path in active if manifest_complete(path)]
    if complete:
        if len(complete) != 1:
            raise RuntimeError(f"multiple complete attempts for {run_dir}")
        for path in active:
            if path != complete[0]:
                _archive_incomplete(path)
        os.replace(complete[0], run_dir)
        return None
    for path in active:
        _archive_incomplete(path)
    return Path(f"{run_dir}.attempt-{next_number}")


def promote_attempt(attempt_dir: Path, run_dir: Path) -> None:
    """Atomically expose an attempt only after its complete manifest is durable."""
    if not manifest_complete(attempt_dir):
        raise RuntimeError(f"attempt has no complete manifest: {attempt_dir}")
    if run_dir.exists():
        raise FileExistsError(f"refusing to replace final run directory: {run_dir}")
    os.replace(attempt_dir, run_dir)


def _queue(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Matrix-driven live runs (the hosted (b) rows on the Mac; one row per cluster job)."""
    all_rows = load_rows(args.matrix)
    if args.input_root:
        relocate_inputs(all_rows, args.input_root)
    by_id = {row["run_id"]: row for row in all_rows}
    missing = [rid for rid in args.run_id if rid not in by_id]
    if missing:
        parser.error(f"unknown run_id(s): {missing}")
    check_binary(args.binary, args.expect_binary_sha256)
    plans = []
    for rid in args.run_id:
        inputs = resolve_row_inputs(by_id[rid], by_id)
        for key in ("config", "schedule", "stream"):
            if not inputs[key].is_file():
                parser.error(f"{rid}: missing {key} input {inputs[key]}")
        if inputs["rng_run"] != 1:
            parser.error(f"{rid}: rng_run must be 1")
        plans.append((rid, inputs))
    wrapper = shlex.split(args.binary_wrapper) if args.binary_wrapper else None

    def execute(rid: str, inputs: dict[str, Any]) -> dict[str, Any]:
        run_dir = args.output_root / rid
        try:
            attempt_dir = prepare_attempt(run_dir)
            if attempt_dir is None:
                return {"run_id": rid, "status": "skip_complete"}
            interpreter = make_interpreter(inputs["interpreter"], args.base_url, args.manifest)
            run_simulator(binary=args.binary, config=inputs["config"], schedule=inputs["schedule"],
                          run_dir=attempt_dir, rng_run=inputs["rng_run"], sim_time=inputs["sim_time"],
                          interpreter=interpreter, replay_dir=None, stream=inputs["stream"],
                          wrapper=wrapper, run_id=rid)
            promote_attempt(attempt_dir, run_dir)
        except Exception as exc:  # noqa: BLE001 - one failed row must not stop the others
            return {"run_id": rid, "status": "failed", "detail": str(exc)[-2000:]}
        return {"run_id": rid, "status": "complete"}

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for future in concurrent.futures.as_completed([pool.submit(execute, *plan) for plan in plans]):
            results.append(future.result())
            print(json.dumps(results[-1]), flush=True)
    return 1 if any(r["status"] == "failed" for r in results) else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ranbench.rq5")
    parser.add_argument("mode", choices=("queue", "live", "replay"))
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--binary-wrapper", help="command prefix for ns-3, e.g. 'apptainer exec --bind ... IMAGE'")
    parser.add_argument("--expect-binary-sha256")
    # queue: matrix rows
    parser.add_argument("--matrix", type=Path)
    parser.add_argument("--input-root")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-id", action="append", default=[])
    parser.add_argument("--jobs", type=int, default=1)
    # live: explicit inputs (smokes)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--schedule", type=Path)
    parser.add_argument("--stream", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--rng-run", type=int, default=1)
    parser.add_argument("--sim-time", type=float)
    parser.add_argument("--interpreter", help="manifest name, or fake-rq5[:seed]")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--base-url")
    # replay: a finished live run directory (its copied inputs, sim time and rng run are reused)
    parser.add_argument("--replay-dir", type=Path)
    args = parser.parse_args(argv)
    wrapper = shlex.split(args.binary_wrapper) if args.binary_wrapper else None

    if args.mode == "queue":
        if not (args.matrix and args.output_root and args.run_id):
            parser.error("queue needs --matrix, --output-root and at least one --run-id")
        return _queue(args, parser)
    if args.run_dir is None:
        parser.error(f"{args.mode} needs --run-dir")
    if args.mode == "live":
        for name in ("config", "schedule", "interpreter"):
            if getattr(args, name) is None:
                parser.error(f"live needs --{name}")
        for path in (args.binary, args.config, args.schedule, args.stream):
            if path is not None and not path.is_file():
                parser.error(f"file not found: {path}")
        result = run_simulator(
            binary=args.binary, config=args.config, schedule=args.schedule, run_dir=args.run_dir,
            rng_run=args.rng_run, sim_time=args.sim_time,
            interpreter=make_interpreter(args.interpreter, args.base_url, args.manifest),
            replay_dir=None, stream=args.stream, wrapper=wrapper,
        )
    else:
        if args.replay_dir is None:
            parser.error("replay needs --replay-dir")
        source = json.loads((args.replay_dir / "manifest.json").read_text(encoding="utf-8"))
        stream = args.replay_dir / "stream.json"
        result = run_simulator(
            binary=args.binary, config=args.replay_dir / "config.json",
            schedule=args.replay_dir / "schedule.csv", run_dir=args.run_dir,
            rng_run=int(source["rng_run"]), sim_time=source["sim_time_s"], interpreter=None,
            replay_dir=args.replay_dir, stream=stream if stream.is_file() else None, wrapper=wrapper,
            run_id=source.get("run_id"),
        )
    print(json.dumps({k: result[k] for k in ("mode", "rq5_ticks", "rq5_decisions", "rq5_skipped",
                                             "rq5_invalid_fields", "wall_time_s")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
