"""Prepare and analyse the full-fidelity C2 pilot controls."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.ranbench.c2 import schedule as sch
from src.ranbench.c2 import stats
from src.ranbench.c2.loader import SLOT_S, RunTraces, load_run
from src.ranbench.c2.outcomes import (
    SLA_CLASSES,
    W_PRIMARY,
    SlotTable,
    actuation,
    intent_outcomes,
    network_series,
    policy_rows_check,
    radio_kpis,
)
from src.ranbench.c2.streams import T0_S, TAIL_S, load_pool, make_stream, write_stream

PILOT_ARMS = ("oracle", "no-update", "fixed-0.1", "fixed-1", "fixed-5")
TRAFFIC_PROFILES: dict[str, dict[str, tuple[float, float]]] = {
    # class -> (offered rate kbps, SLA target). Only these two traffic fields change.
    "balanced-a": {
        "video": (4000.0, 3200.0),
        "xr": (1200.0, 20.0),
        "iot": (300.0, 240.0),
        "be": (2000.0, 10000.0),
    },
    "balanced-b": {
        "video": (3000.0, 2400.0),
        "xr": (900.0, 25.0),
        "iot": (300.0, 240.0),
        "be": (1500.0, 1200.0),
    },
    "balanced-c": {
        "video": (1200.0, 900.0),
        "xr": (400.0, 35.0),
        "iot": (300.0, 240.0),
        "be": (500.0, 400.0),
    },
}


def _truncate_stream(stream: dict[str, Any], sim_time_s: float, block_s: float) -> dict[str, Any]:
    out = json.loads(json.dumps(stream))
    # Keep the registered 10 s tail so every W <= 10 s window is observable.
    last_issue = sim_time_s - TAIL_S
    out["events"] = [event for event in out["events"] if event["t_issue_s"] <= last_issue]
    length = sim_time_s - T0_S - TAIL_S
    out["meta"].update({
        "key": f"pilot-base-{sim_time_s:g}s",
        "run_length_s": length,
        "sim_time_s": sim_time_s,
        "n_blocks": max(1, math.floor(length / block_s + 1e-9)),
        "n_events": len(out["events"]),
    })
    return out


def prepare(out: Path, base_config: Path, pool_path: Path, profile: str,
            sim_time_s: float, put_ack_median_s: float, block_s: float = 10.0,
            allow_a1_put_ack_override: bool = False) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite pilot directory: {out}")
    if profile not in TRAFFIC_PROFILES:
        raise ValueError(f"unknown traffic profile {profile}")
    if put_ack_median_s < 0:
        raise ValueError("A1 PUT-ack median must be non-negative")
    out.mkdir(parents=True)
    pool, pool_sha = load_pool(pool_path)
    stream = make_stream(pool, 0.3, block_s, "pilot-base", pool_sha256=pool_sha)
    if sim_time_s < stream["meta"]["sim_time_s"]:
        stream = _truncate_stream(stream, sim_time_s, block_s)
    stream_path = out / "stream.json"
    stream_sha = write_stream(stream, stream_path)

    config = json.loads(base_config.read_text(encoding="utf-8"))
    changed: dict[str, dict[str, float]] = {}
    for cls, (rate, target) in TRAFFIC_PROFILES[profile].items():
        spec = config["traffic"]["classes"][cls]
        spec["rate_kbps"] = rate
        spec["target_value"] = target
        changed[cls] = {"rate_kbps": rate, "target_value": target}
    config["simulation"]["sim_time_s"] = stream["meta"]["sim_time_s"]
    config_path = out / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    arms_dir = out / "arms"
    d_a1 = sch.delta_a1(put_ack_median_s, allow_a1_put_ack_override)
    for name in PILOT_ARMS:
        if name == "oracle":
            arm = sch.build_arm(stream, "oracle")
        elif name == "no-update":
            arm = sch.build_arm(stream, "no_update")
        else:
            delay = float(name.removeprefix("fixed-"))
            arm = sch.build_arm(stream, "fixed", "E", fixed_d=delay, d_a1=d_a1)
        sch.write_arm(arm, arms_dir / name)

    spec = {
        "profile": profile,
        "traffic_changes_only": changed,
        "sim_time_s": stream["meta"]["sim_time_s"],
        "block_s": block_s,
        "a1_put_ack_median_s": put_ack_median_s,
        "a1_put_ack_override": (
            allow_a1_put_ack_override
            and abs(put_ack_median_s - sch.PUT_ACK_MEDIAN_S) > 1e-12
        ),
        "delta_a1_s": d_a1,
        "delta_e2_s": sch.DELTA_E2_S,
        "n_events": len(stream["events"]),
        "stream_sha256": stream_sha,
        "rng_run": 1,
        "arms": list(PILOT_ARMS),
    }
    (out / "pilot-spec.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    return spec


def _rows(path: Path, tolerate_torn_tail: bool = False) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if text and not text.endswith("\n"):
        if not tolerate_torn_tail:
            raise ValueError(f"{path}: last line has no newline (run still writing or killed)")
        text = text[: text.rfind("\n") + 1]
    return list(csv.DictReader(text.splitlines()))


def silent_periods(run_dir: Path, min_s: float = 2.0, pre_s: float = 0.2, require_event: bool = True,
                   tolerate_torn_tail: bool = False) -> list[tuple[int, float, float]]:
    """Periods of >= min_s with sends but zero received bytes; by default only handover/RLF-associated ones."""
    rows = lambda name: _rows(run_dir / name, tolerate_torn_tail)  # noqa: E731
    rx = {(row["t_slot"], row["imsi"]): int(row["rx_bytes"]) for row in rows("ue_slots.csv")}
    tx = {(row["t_slot"], row["imsi"]): int(row["tx_bytes"]) for row in rows("tx_trace.csv")}
    events: dict[str, list[float]] = defaultdict(list)
    for row in rows("handover.csv"):
        for key in ("t_start", "t_end_ok"):
            if row[key]:
                events[row["imsi"]].append(float(row[key]))
    for row in rows("rlf.csv"):
        events[row["imsi"]].append(float(row["t"]))
    slots = sorted({time for time, _ in rx}, key=float)
    imsis = sorted({imsi for _, imsi in rx}, key=int)
    periods: list[tuple[int, float, float]] = []
    for imsi in imsis:
        quiet: list[str] = []
        for time in slots + [None]:
            is_quiet = time is not None and tx.get((time, imsi), 0) > 0 and rx.get((time, imsi), 0) == 0
            if is_quiet:
                quiet.append(time)
                continue
            if quiet:
                start, end = float(quiet[0]) - 0.1, float(quiet[-1])
                near_event = any(start - pre_s <= event <= end for event in events[imsi])
                if end - start >= min_s - 1e-9 and (near_event or not require_event):
                    periods.append((int(imsi), round(start, 3), round(end, 3)))
                quiet = []
    return periods


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_complete(pilot_dir: Path, run_dir: Path, arm: str, sim_time_s: float) -> None:
    """Fail closed unless the runner finished this arm on exactly the pilot's config and schedule."""
    path = run_dir / "manifest.json"
    if not path.is_file():
        raise RuntimeError(f"{arm}: {path} missing (run incomplete or failed)")
    m = json.loads(path.read_text(encoding="utf-8"))
    problems = []
    if m.get("complete") is not True:
        problems.append("manifest not complete")
    if m.get("sim_time_s") is None or abs(float(m["sim_time_s"]) - sim_time_s) > 1e-9:
        problems.append(f"sim_time_s {m.get('sim_time_s')} != {sim_time_s}")
    if m.get("config_sha256") != _sha256(pilot_dir / "config.json"):
        problems.append("config differs from pilot config.json")
    if m.get("schedule_sha256") != _sha256(pilot_dir / "arms" / arm / "schedule.csv"):
        problems.append("schedule differs from pilot arm schedule")
    if problems:
        raise RuntimeError(f"{arm}: " + "; ".join(problems))


def _silent_share(periods: list[tuple[int, float, float]], n_ue: int, t_from: float, t_to: float) -> float:
    """Share of UE-time in (t_from, t_to] inside the given silent periods."""
    covered = sum(max(0.0, min(end, t_to) - max(start, t_from)) for _, start, end in periods)
    return covered / (n_ue * (t_to - t_from)) if n_ue and t_to > t_from else float("nan")


def control_autocorr_times(
    traces: dict[str, RunTraces], horizon: float
) -> dict[str, float]:
    """Measure the five control-arm SLA autocorrelation times used by D-10."""
    autocorr: dict[str, float] = {}
    for name in PILOT_ARMS:
        slot_table = SlotTable(traces[name])
        times, series = network_series(traces[name], slot_table)
        series = series[(times > T0_S + 1e-9) & (times <= horizon + 1e-9)]
        autocorr[name] = stats.autocorr_time_s(series, SLOT_S)
    return autocorr


def analyse_controls(
    stream: dict[str, Any], traces: dict[str, RunTraces],
    events_by_arm: dict[str, list[dict[str, Any]]], horizon: float,
    block_assignments: dict[str, dict[str, Any]],
    autocorr_time_by_arm_s: dict[str, float] | None = None,
) -> dict[str, Any]:
    """The shared five-arm C-1/C-2 calculation used by pilot and production runs."""
    if not block_assignments:
        raise ValueError("at least one explicit block assignment is required")
    n_events = len(stream["events"])
    for label, assignment in block_assignments.items():
        if len(assignment["blocks"]) != n_events:
            raise ValueError(f"{label}: block assignment has the wrong number of events")
    outcomes: dict[str, list[dict[str, Any]]] = {}
    slot_tables: dict[str, SlotTable] = {}
    for name in PILOT_ARMS:
        slot_table = SlotTable(traces[name])
        slot_tables[name] = slot_table
        outcomes[name] = intent_outcomes(traces[name], events_by_arm[name], slot_table, t_end=horizon)
        for row in outcomes[name]:
            if row["t_issue_s"] + W_PRIMARY > horizon + 1e-9:
                row["net_W2"] = row["aff_W2"] = None
    autocorr = dict(autocorr_time_by_arm_s or control_autocorr_times(traces, horizon))
    block_lengths = {name: max(stats.MIN_BLOCK_S, value) for name, value in autocorr.items()}

    net = {name: stats.as_array(rows, "net_W2") for name, rows in outcomes.items()}
    fixed = {delay: net[f"fixed-{delay:g}"] for delay in (0.1, 1.0, 5.0)}
    # All-intent affected-class versions (descriptive): aff_W2 unsigned; BE intents have no SLA (undefined).
    aff_all = {name: stats.as_array(rows, "aff_W2") for name, rows in outcomes.items()}
    # Deviation D-5 (the eligibility controls): affected class, actuating intents, signed by intended direction.
    oracle_events = events_by_arm["oracle"]
    act = actuation(oracle_events, traces["oracle"].clusters, traces["oracle"].config["priorities"])
    direction = np.array([row["direction"] for row in act], dtype=float)
    controls_by_block: dict[str, dict[str, dict[str, Any]]] = {}
    for label, assignment in block_assignments.items():
        blocks, counts = assignment["blocks"], assignment["counts"]
        c1 = stats.c1_policy_matters(net["no-update"], net["oracle"], blocks, counts)
        c2 = stats.c2_latency_matters(fixed, blocks, counts)
        c1_aff = stats.c1_policy_matters(aff_all["no-update"], aff_all["oracle"], blocks, counts)
        c2_aff = stats.c2_latency_matters(
            {d: aff_all[f"fixed-{d:g}"] for d in (0.1, 1.0, 5.0)}, blocks, counts)
        d5 = stats.d5_controls(
            aff_all["no-update"], aff_all["oracle"],
            {d: aff_all[f"fixed-{d:g}"] for d in (0.1, 1.0, 5.0)}, direction, blocks, counts)
        controls_by_block[label] = {
            "C-1": d5["C-1"], "C-2": d5["C-2"],
            "C-1-network-descriptive": c1, "C-2-network-descriptive": c2,
            "C-1-all-intent-affected-descriptive": c1_aff,
            "C-2-all-intent-affected-descriptive": c2_aff,
        }
    primary_label = "primary" if "primary" in block_assignments else next(iter(block_assignments))
    primary_assignment = block_assignments[primary_label]
    return {
        "controls": controls_by_block[primary_label],
        "controls_by_block": controls_by_block,
        "actuation": {
            "rule": "D-5: SLA-class true install changes the oracle-arm base level in scope; "
                    "direction +1 raises priority (lower m_priority), -1 lowers it",
            "intents": act,
            "oracle_policy_rows_check": policy_rows_check(
                traces["oracle"].enforcement, oracle_events, traces["oracle"].clusters,
                traces["oracle"].config["priorities"]),
        },
        "autocorr_time_by_arm_s": autocorr,
        "block_length_s": max(block_lengths.values()),
        "block_length_by_arm_s": block_lengths,
        "blocks": primary_assignment["blocks"],
        "counts": primary_assignment["counts"],
        "outcomes": outcomes,
        "slot_tables": slot_tables,
    }


def iot_gap(
    no_update: np.ndarray, oracle: np.ndarray, assignments: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """IoT affected-class gap (no-update - oracle) under B and the registered 10 s blocking (D-10).

    Secondary contrast, in no Holm family: resolved only if the CI excludes 0 and the raw p < 0.05
    under both blockings."""
    primary = stats.strip(stats.gap(
        no_update, oracle, assignments["primary"]["blocks"], assignments["primary"]["counts"]))
    sensitivity = stats.strip(stats.gap(
        no_update, oracle, assignments["sensitivity_10s"]["blocks"],
        assignments["sensitivity_10s"]["counts"]))
    pos, neg = stats.resolved_both(primary, sensitivity, "p_raw")
    return {
        "aff_W2_no_update_minus_oracle": primary,
        "aff_W2_no_update_minus_oracle_10s": sensitivity,
        "aff_W2_no_update_minus_oracle_resolved_both_raw_p": bool(pos or neg),
    }


def analyse(pilot_dir: Path, runs_dir: Path, allow_partial: bool = False) -> dict[str, Any]:
    """Controls C-1/C-2 and the calibration diagnostics for one candidate's five arm runs.

    allow_partial=True reads in-flight or killed runs: it skips the manifest gate, drops torn last lines, cuts every
    arm to the common simulated horizon and leaves intents whose W = 2 s window passes that horizon undefined.
    Its output is a progress look only and is labelled as such."""
    stream = json.loads((pilot_dir / "stream.json").read_text(encoding="utf-8"))
    config = json.loads((pilot_dir / "config.json").read_text(encoding="utf-8"))
    sim_time_s = float(config["simulation"]["sim_time_s"])
    traces = {}
    for name in PILOT_ARMS:
        if not allow_partial:
            check_complete(pilot_dir, runs_dir / name, name, sim_time_s)
        traces[name] = load_run(runs_dir / name, tolerate_torn_tail=allow_partial)
    horizon = sim_time_s
    if allow_partial:
        horizon = min(float(tr.ue_slots["t"].max()) if len(tr.ue_slots) else 0.0 for tr in traces.values())
        horizon = min(horizon, sim_time_s)
        for tr in traces.values():
            tr.ue_slots = tr.ue_slots[tr.ue_slots["t"] <= horizon + 1e-9]
            tr.cell_prb = tr.cell_prb[tr.cell_prb["t"] <= horizon + 1e-9]
    if horizon <= T0_S:
        raise RuntimeError(f"common horizon {horizon} s does not pass t0 = {T0_S} s")

    events_by_arm = {
        name: json.loads((pilot_dir / "arms" / name / "events.json").read_text(encoding="utf-8"))["events"]
        for name in PILOT_ARMS
    }
    autocorr = control_autocorr_times(traces, horizon)
    d10_assignments = stats.d10_block_assignments_from_stream(stream, horizon, autocorr)
    assignments = {
        "primary": d10_assignments["primary"],
        "sensitivity_10s": d10_assignments["sensitivity_10s"],
    }
    control_analysis = analyse_controls(
        stream, traces, events_by_arm, horizon, assignments, autocorr)
    outcomes = control_analysis["outcomes"]
    kpis: dict[str, dict[str, Any]] = {}
    sla_by_class: dict[str, dict[str, float | None]] = {}
    silent: dict[str, list[tuple[int, float, float]]] = {}
    silent_share: dict[str, dict[str, float]] = {}
    prb: dict[str, dict[str, Any]] = {}
    for name, tr in traces.items():
        run_dir = runs_dir / name
        slot_table = control_analysis["slot_tables"][name]
        kpis[name] = radio_kpis(tr, T0_S, horizon, slots=slot_table)
        post = (slot_table.t > T0_S + 1e-9) & (slot_table.t <= horizon + 1e-9)
        sla_by_class[name] = {"network": float(slot_table.miss[post].mean())} | {
            cls: (float(slot_table.miss[post & (slot_table.cls == cls)].mean())
                  if (post & (slot_table.cls == cls)).any() else None)
            for cls in SLA_CLASSES
        }
        per_cell = kpis[name]["prb_util_per_cell"]
        prb[name] = {"mean": kpis[name]["prb_util_mean"], "max": max(per_cell.values()) if per_cell else None,
                     "n_cells_ge_0p95": sum(v >= 0.95 for v in per_cell.values()), "per_cell": per_cell}
        silent[name] = silent_periods(run_dir, tolerate_torn_tail=allow_partial)
        n_ue = len(np.unique(slot_table.imsi))
        silent_share[name] = {
            "ho_rlf_associated": _silent_share(silent[name], n_ue, T0_S, horizon),
            "any_cause": _silent_share(silent_periods(run_dir, require_event=False,
                                                      tolerate_torn_tail=allow_partial), n_ue, T0_S, horizon),
        }

    is_iot = np.array([row["cls"] == "iot" for row in outcomes["oracle"]])
    aff = {name: np.where(is_iot, stats.as_array(rows, "aff_W2"), np.nan) for name, rows in outcomes.items()}
    iot_spec = config["traffic"]["classes"]["iot"]
    iot = {
        "rate_kbps": iot_spec["rate_kbps"], "target_value": iot_spec["target_value"],
        "target_type": iot_spec["target_type"],
        "slot_miss_share_by_arm": {name: v["iot"] for name, v in sla_by_class.items()},
        "n_iot_intents": int(is_iot.sum()),
        **iot_gap(aff["no-update"], aff["oracle"], assignments),
    }
    silent_sets = {name: set(periods) for name, periods in silent.items()}
    identical = len({tuple(sorted(periods)) for periods in silent_sets.values()}) == 1
    result = {
        "status": "PARTIAL progress look, not a verdict" if allow_partial else "complete",
        "pilot_dir": str(pilot_dir),
        "runs_dir": str(runs_dir),
        "horizon_s": horizon,
        "n_blocks": assignments["primary"]["n_blocks"],
        "n_blocks_10s": assignments["sensitivity_10s"]["n_blocks"],
        "block_s": assignments["primary"]["block_s"],
        "block_10s_s": assignments["sensitivity_10s"]["block_s"],
        "tau_s": d10_assignments["tau_s"],
        "warnings": ([f"n_blocks = {assignments['primary']['n_blocks']}: the block bootstrap has at most n^n distinct "
                      "resamples, so C-1/C-2 resolution is indicative only"]
                     if assignments["primary"]["n_blocks"] < 5 else []),
        "controls": control_analysis["controls"],
        "controls_10s": control_analysis["controls_by_block"]["sensitivity_10s"],
        "actuation": control_analysis["actuation"],
        "sla_violation_share_by_class": sla_by_class,
        "iot_responsiveness": iot,
        "prb_utilisation": prb,
        "autocorr_time_by_arm_s": control_analysis["autocorr_time_by_arm_s"],
        "block_length_s": control_analysis["block_length_s"],
        "block_length_by_arm_s": control_analysis["block_length_by_arm_s"],
        "kpis": kpis,
        "silent_periods": {name: [list(period) for period in periods] for name, periods in silent.items()},
        "silent_share": silent_share,
        "silent_periods_identical": identical,
        "silent_exclusion_proposal": (
            "exclude the identical handover/RLF-associated UE-periods in every arm"
            if identical and any(silent.values()) else "do not exclude: periods are absent or differ across arms"
        ),
        "n_events": len(stream["events"]),
        "bootstrap_resamples": stats.N_RESAMPLES,
        "bootstrap_seed": stats.SEED,
    }
    out_name = "analysis.partial.json" if allow_partial else "analysis.json"
    (pilot_dir / out_name).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare/analyse the C2 full-fidelity pilot")
    sub = parser.add_subparsers(dest="cmd", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--out", required=True)
    prep.add_argument("--base-config", required=True)
    prep.add_argument("--pool", required=True)
    prep.add_argument("--profile", choices=sorted(TRAFFIC_PROFILES), required=True)
    prep.add_argument("--sim-time", type=float, required=True)
    prep.add_argument("--a1-put-ack-median-s", type=float, required=True)
    prep.add_argument("--allow-a1-put-ack-override", action="store_true")
    prep.add_argument("--block", type=float, default=10.0)
    ana = sub.add_parser("analyse")
    ana.add_argument("--pilot-dir", required=True)
    ana.add_argument("--runs-dir", required=True)
    ana.add_argument("--allow-partial", action="store_true",
                     help="progress look at in-flight runs (no manifest gate, common horizon); never a verdict")
    args = parser.parse_args(argv)
    if args.cmd == "prepare":
        result = prepare(Path(args.out), Path(args.base_config), Path(args.pool), args.profile,
                         args.sim_time, args.a1_put_ack_median_s, args.block,
                         args.allow_a1_put_ack_override)
    else:
        result = analyse(Path(args.pilot_dir), Path(args.runs_dir), args.allow_partial)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
