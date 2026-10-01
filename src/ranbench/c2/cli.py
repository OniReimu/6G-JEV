"""CLI for the C2 Python side: stream -> schedule per arm -> (ns-3) -> outcomes; C-4 check.

  python -m src.ranbench.c2.cli stream   --pool POOL --rate R --speed S --block B --out stream.json
  python -m src.ranbench.c2.cli schedule --stream stream.json --arm L --mode E --records ledger.jsonl \\
                                         --model M --a1-put-ack-ms X --out ARM_DIR
  python -m src.ranbench.c2.cli outcomes --run RUN_DIR --events ARM_DIR/events.json --out outcomes.json
  python -m src.ranbench.c2.cli outcomes --run RUN_DIR --from-schedule   (smoke: schedule rows as oracle intents)
  python -m src.ranbench.c2.cli c4 RUN_A RUN_B
  python -m src.ranbench.c2.cli matrix --input-root INPUTS --out matrix.csv
  python -m src.ranbench.c2.cli materialize --matrix matrix.csv --input-root INPUTS ...
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from src.ranbench.c2 import schedule as sch
from src.ranbench.c2.assignment import DEFAULT_SLOTS, assign, write_assignment
from src.ranbench.c2.loader import load_run
from src.ranbench.c2.materialize import materialize, read_matrix
from src.ranbench.c2.matrix import (
    DEFAULT_PLATFORM_MULTIPLIERS,
    generate_matrix,
    totals,
    write_matrix,
)
from src.ranbench.c2.outcomes import W_PRIMARY, SlotTable, intent_outcomes, radio_kpis
from src.ranbench.c2.stats import c4_pairing
from src.ranbench.c2.streams import (
    T0_S,
    load_pool,
    load_stream,
    make_stream,
    write_stream,
)


def _events_from_schedule(run_dir: Path) -> list[dict]:
    with open(run_dir / "schedule.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [{"j": j, "intent_id": f"sched_{j}", "t_issue_s": float(r["t_s"]), "e_s": float(r["t_s"]),
             "latency_s": 0.0, "correct": True, "true_install": [r["scope"], r["class"], r["priority"]],
             "install": [r["scope"], r["class"], r["priority"]], "in_schedule": True} for j, r in enumerate(rows)]


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))


def _finite(v):
    return None if isinstance(v, float) and not np.isfinite(v) else v


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="ranbench.c2")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stream")
    s.add_argument("--pool", required=True)
    s.add_argument("--rate", type=float, required=True)
    s.add_argument("--speed", type=float, required=True)
    s.add_argument("--block", type=float, required=True)
    s.add_argument("--out", required=True)
    s = sub.add_parser("schedule")
    s.add_argument("--stream", required=True)
    s.add_argument("--arm", choices=sch.ARMS, required=True)
    s.add_argument("--mode", choices=sch.MODES, default="E")
    s.add_argument("--records")
    s.add_argument("--model")
    s.add_argument("--fixed-d", type=float)
    s.add_argument("--a1-put-ack-ms", type=float, help="median A1 PUT-acknowledge time from C3")
    s.add_argument("--allow-a1-put-ack-override", action="store_true",
                   help="explicitly allow a PUT-ack median other than the pinned C3 value")
    s.add_argument("--out", required=True)
    s = sub.add_parser("outcomes")
    s.add_argument("--run", required=True)
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--events")
    g.add_argument("--from-schedule", action="store_true")
    s.add_argument("--t0", type=float, default=T0_S)
    s.add_argument("--tolerate-torn-tail", action="store_true")
    s.add_argument("--out")
    s = sub.add_parser("c4")
    s.add_argument("run_a")
    s.add_argument("run_b")
    s = sub.add_parser("matrix")
    s.add_argument("--input-root", required=True)
    s.add_argument("--block", type=float, default=10.0)
    s.add_argument("--out", required=True)
    s.add_argument("--m5-multiplier", type=float, default=DEFAULT_PLATFORM_MULTIPLIERS["mac_m5_max"])
    s.add_argument("--l40-multiplier", type=float, default=DEFAULT_PLATFORM_MULTIPLIERS["l40_cpu"])
    s.add_argument("--load-trace", action="append", default=[], metavar="MODEL@RATE=TRACE",
                   help="complete RQ3 trace; derives exact replay duration")
    s = sub.add_parser("assign")
    s.add_argument("--matrix", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--slots", default=",".join(f"{k}={v}" for k, v in DEFAULT_SLOTS.items()),
                   help="PLATFORM=SLOTS,...; a platform is one binary on one CPU type")
    s.add_argument("--multiplier", action="append", default=[], metavar="PLATFORM=X",
                   help="wall-time multiplier vs M4 for a platform without a matrix column "
                        "(e.g. cluster_a after its probe)")
    s.add_argument("--pin", action="append", default=[], metavar="GLOB=PLATFORM",
                   help="place matching design points on PLATFORM before greedy assignment; repeatable")
    s = sub.add_parser("materialize")
    s.add_argument("--matrix", required=True)
    s.add_argument("--input-root", required=True)
    s.add_argument("--base-config", required=True,
                   help="pilot-confirmed config containing the chosen offered loads and targets")
    s.add_argument("--pool", required=True)
    s.add_argument("--block", type=float, required=True)
    s.add_argument("--a1-put-ack-median-s", type=float, required=True)
    s.add_argument("--allow-a1-put-ack-override", action="store_true",
                   help="explicitly allow and record a value other than the pinned C3 median")
    s.add_argument("--records", action="append", default=[], metavar="MODEL=LEDGER",
                   help="selected C1 ledger; repeat once per available interpreter")
    s.add_argument("--load-trace", action="append", default=[], metavar="MODEL@RATE=TRACE",
                   help="complete RQ3 trace; repeat once per available interpreter/rate")
    a = ap.parse_args(argv)

    if a.cmd == "stream":
        pool, pool_sha = load_pool(a.pool)
        stream = make_stream(pool, a.rate, a.block, key=f"rate={a.rate:g},speed={a.speed:g}", pool_sha256=pool_sha)
        sha = write_stream(stream, a.out)
        print(json.dumps(stream["meta"] | {"sha256": sha}, indent=1))
    elif a.cmd == "schedule":
        stream = load_stream(a.stream)
        records = sch.load_records(a.records, a.model) if a.records else None
        d_a1 = (sch.delta_a1(a.a1_put_ack_ms / 1000.0, a.allow_a1_put_ack_override)
                if a.a1_put_ack_ms is not None else None)
        arm = sch.build_arm(stream, a.arm, a.mode, records, a.fixed_d, d_a1=d_a1)
        arm["meta"]["model"] = a.model
        sch.write_arm(arm, a.out)
        print(f"{a.arm}: {len(arm['schedule'])} schedule rows, "
              f"{sum(e['correct'] for e in arm['events'])}/{len(arm['events'])} correct -> {a.out}")
    elif a.cmd == "outcomes":
        run = Path(a.run)
        tr = load_run(run, tolerate_torn_tail=a.tolerate_torn_tail)
        events = _events_from_schedule(run) if a.from_schedule else json.loads(Path(a.events).read_text())["events"]
        t_end = float(tr.ue_slots["t"].max())  # last complete slot (== sim end for a finished run)
        st = SlotTable(tr)
        rows = intent_outcomes(tr, events, st, t_end=t_end)
        kpis = radio_kpis(tr, a.t0, t_end, st)
        out = {"run": str(run), "t_end_s": t_end, "intents": [{k: _finite(v) for k, v in r.items()} for r in rows],
               "kpis": kpis}
        if a.out:
            Path(a.out).write_text(json.dumps(out, indent=1, default=_json_default) + "\n")
        w = f"W{W_PRIMARY:g}"
        f4 = lambda v: "-" if v is None else f"{v:.4f}"
        for r in rows:
            print(f"j={r['j']:>3} t={r['t_issue_s']:7.3f} {r['scope']:>13} {r['cls']:>5} "
                  f"aff_{w}={f4(r[f'aff_{w}'])} net_{w}={f4(r[f'net_{w}'])} "
                  f"aff_life={f4(r['aff_life'])} S={r['S_ue_s']:.2f}")
        print(json.dumps({k: v for k, v in kpis.items() if k != "prb_util_per_cell"}, indent=1,
                         default=_json_default))
    elif a.cmd == "c4":
        res = c4_pairing(a.run_a, a.run_b)
        print(json.dumps(res, indent=1))
        sys.exit(0 if res["pass"] else 1)
    elif a.cmd == "matrix":
        factors = dict(DEFAULT_PLATFORM_MULTIPLIERS)
        factors["mac_m5_max"] = a.m5_multiplier
        factors["l40_cpu"] = a.l40_multiplier
        trace_paths = dict(item.split("=", 1) for item in a.load_trace)
        rows = generate_matrix(a.input_root, block_s=a.block, platform_multipliers=factors,
                               rq3_trace_paths=trace_paths)
        write_matrix(rows, a.out)
        print(json.dumps(totals(rows), indent=1))
    elif a.cmd == "assign":
        with Path(a.matrix).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        slots = {key: int(value) for key, value in (item.split("=", 1) for item in a.slots.split(","))}
        extra = {key: float(value) for key, value in (item.split("=", 1) for item in a.multiplier)}
        pins = [tuple(item.split("=", 1)) for item in a.pin]
        print(json.dumps(write_assignment(assign(rows, slots, extra, pins), a.out, rows), indent=1))
    elif a.cmd == "materialize":
        record_paths = dict(item.split("=", 1) for item in a.records)
        trace_paths = dict(item.split("=", 1) for item in a.load_trace)
        result = materialize(read_matrix(a.matrix), a.input_root, a.base_config, a.pool,
                             a.block, a.a1_put_ack_median_s, record_paths, trace_paths,
                             a.allow_a1_put_ack_override)
        print(json.dumps({k: v for k, v in result.items() if k not in {"files", "row_status"}}, indent=1))


if __name__ == "__main__":
    main()
