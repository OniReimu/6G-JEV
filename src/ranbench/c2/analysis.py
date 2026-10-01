"""Production C2 analysis driver for EXP-2026-003."""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.ranbench.c2 import stats
from src.ranbench.c2.eligibility import d10_eligibility_flags
from src.ranbench.c2.loader import load_run, pairing_identical
from src.ranbench.c2.matrix import INTERPRETERS, RQ2_RATES, RQ2_SPEEDS
from src.ranbench.c2.outcomes import (
    W_PRIMARY, SlotTable, class_jain, intent_outcomes, radio_kpis, stale_policy_exposure,
)
from src.ranbench.c2.pilot import PILOT_ARMS, analyse_controls, control_autocorr_times
from src.ranbench.c2.streams import T0_S
from src.ranbench.c2.verify import sha256_file, verify_run

BASE_POINT = "dp_r0p3_s30_u5"
D7_MARKER = re.compile(r"\bEXP2026003_PATCH_D7\b")
D7B_MARKER = re.compile(r"\bEXP2026003_PATCH_D7B\b")


@dataclass(frozen=True)
class LocatedRun:
    run_dir: Path
    platform_root: Path


def _matrix_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    run_ids = [row.get("run_id", "") for row in rows]
    if not rows or any(not run_id for run_id in run_ids):
        raise ValueError(f"matrix is empty or has a row without run_id: {path}")
    if len(run_ids) != len(set(run_ids)):
        raise ValueError(f"matrix has duplicate run_id values: {path}")
    return rows


def discover_run_dirs(rows: Iterable[dict[str, str]], roots: list[Path]) -> dict[str, LocatedRun]:
    """Locate every present matrix run, rejecting ambiguous copies across roots."""
    if not roots:
        raise ValueError("at least one runs root is required")
    found: dict[str, LocatedRun] = {}
    for row in rows:
        run_id = row["run_id"]
        matches = [(root, root / run_id) for root in roots if (root / run_id).is_dir()]
        if len(matches) > 1:
            paths = [str(path) for _, path in matches]
            raise RuntimeError(f"{run_id}: run directory exists under multiple --runs-root values: {paths}")
        if matches:
            root, path = matches[0]
            found[run_id] = LocatedRun(path, root)
    return found


def _verify(
    rows: list[dict[str, str]], located: dict[str, LocatedRun], allow_incomplete: bool
) -> dict[str, tuple[str, list[str]]]:
    """verify_run every matrix run. A run that is absent (MISSING: no directory; INCOMPLETE: no manifest.json
    yet) stops the analysis unless allow_incomplete, which leaves it to the dependency rule in analyse(). A run
    that is present and fails verification (FAIL, including a manifest that is not complete) always stops it."""
    status: dict[str, tuple[str, list[str]]] = {}
    bad: list[str] = []
    unavailable: list[str] = []
    for row in rows:
        run_id = row["run_id"]
        run_dir = located[run_id].run_dir if run_id in located else Path("/__missing__") / run_id
        result = verify_run(run_dir, row, None)
        state, problems = result
        status[run_id] = result
        if state == "FAIL":
            bad.append(f"{run_id}: {'; '.join(problems)}")
        elif state != "PASS":
            unavailable.append(f"{run_id}: {state}: {'; '.join(problems)}")
    if bad:
        raise RuntimeError("invalid completed run(s):\n" + "\n".join(bad))
    if unavailable and not allow_incomplete:
        raise RuntimeError("required run(s) missing or incomplete:\n" + "\n".join(unavailable))
    return status


def _one(rows: list[dict[str, str]], **want: str) -> dict[str, str]:
    matches = [row for row in rows if all(row.get(key) == value for key, value in want.items())]
    if len(matches) != 1:
        detail = ", ".join(f"{key}={value}" for key, value in want.items())
        raise ValueError(f"expected one matrix row for {detail}, found {len(matches)}")
    return matches[0]


def _events(row: dict[str, str]) -> list[dict[str, Any]]:
    path = Path(row["schedule_file"]).with_suffix(".events.json")
    if not path.is_file():
        raise RuntimeError(f"{row['run_id']}: events missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))["events"]


def _manifest(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


def _available(
    required: list[dict[str, str]], status: dict[str, tuple[str, list[str]]],
    skipped: list[dict[str, Any]], analysis: str, design_point: str,
    interpreter: str | None = None, **extra: str,
) -> bool:
    missing = [
        {"run_id": row["run_id"], "status": status[row["run_id"]][0],
         "detail": "; ".join(status[row["run_id"]][1])}
        for row in required if status[row["run_id"]][0] != "PASS"
    ]
    if not missing:
        return True
    entry: dict[str, Any] = {
        "analysis": analysis,
        "design_point": design_point,
        "required_runs_unavailable": missing,
    }
    if interpreter is not None:
        entry["interpreter"] = interpreter
    skipped.append(entry | extra)
    return False


def _point_meta(design_point: str, rows: list[dict[str, str]]) -> dict[str, Any]:
    row = rows[0]
    return {
        "design_point": design_point,
        "rqs": row["rqs"],
        "rate_per_s": float(row["rate_per_s"]),
        "speed_kmh": float(row["speed_kmh"]),
        "ues_per_cell": int(row["ues_per_cell"]),
    }


def _control_analysis(
    design_point: str, rows: list[dict[str, str]], located: dict[str, LocatedRun]
) -> dict[str, Any]:
    controls = {arm: _one(rows, arm=arm, interpreter="control") for arm in PILOT_ARMS}
    run_dirs = {arm: located[row["run_id"]].run_dir for arm, row in controls.items()}
    manifests = {arm: _manifest(run_dir) for arm, run_dir in run_dirs.items()}
    horizons = {float(manifest["sim_time_s"]) for manifest in manifests.values()}
    if len(horizons) != 1:
        raise RuntimeError(f"{design_point}: control runs have different simulated horizons")
    stream_shas = {manifest.get("stream_sha256") for manifest in manifests.values()}
    if len(stream_shas) != 1:
        raise RuntimeError(f"{design_point}: control runs used different streams")
    stream = json.loads((run_dirs["oracle"] / "stream.json").read_text(encoding="utf-8"))
    traces = {arm: load_run(run_dir) for arm, run_dir in run_dirs.items()}
    events_by_arm = {arm: _events(row) for arm, row in controls.items()}
    horizon = horizons.pop()
    autocorr = control_autocorr_times(traces, horizon)
    d10_assignments = stats.d10_block_assignments_from_stream(stream, horizon, autocorr)
    assignments = {
        "primary": d10_assignments["primary"],
        "sensitivity_10s": d10_assignments["sensitivity_10s"],
    }
    result = analyse_controls(
        stream, traces, events_by_arm, horizon, assignments, autocorr)
    controls_10s = result["controls_by_block"]["sensitivity_10s"]
    return {
        "stream": stream,
        "assignments": assignments,
        "controls": result["controls"],
        "controls_10s": controls_10s,
        "actuation": result["actuation"],
        "tau_s": d10_assignments["tau_s"],
        "block_length_s": result["block_length_s"],
        "block_pass": True,
        **d10_eligibility_flags(result["controls"], controls_10s),
    }


def _outcome_arrays(
    row: dict[str, str], located: dict[str, LocatedRun], stream: dict[str, Any]
) -> dict[str, np.ndarray]:
    trace = load_run(located[row["run_id"]].run_dir)
    events = _events(row)
    stream_ids = [event["j"] for event in stream["events"]]
    event_ids = [event["j"] for event in events]
    if event_ids != stream_ids:
        raise RuntimeError(f"{row['run_id']}: event order differs from its design-point stream")
    outcomes = intent_outcomes(trace, events, SlotTable(trace), t_end=trace.sim_time_s)
    for outcome in outcomes:
        if outcome["t_issue_s"] + W_PRIMARY > trace.sim_time_s + 1e-9:
            outcome["aff_W2"] = outcome["net_W2"] = None
    return {
        "aff": stats.as_array(outcomes, "aff_W2"),
        "net": stats.as_array(outcomes, "net_W2"),
    }


def _stat_columns(prefix: str, value: dict[str, Any] | None, include_p: bool = False) -> dict[str, Any]:
    keys = ("point", "ci_low", "ci_high", "n")
    out = {f"{prefix}_{key}": None if value is None else value.get(key) for key in keys}
    if include_p:
        out[f"{prefix}_p_raw"] = None if value is None else value.get("p_raw")
    return out


def _block_columns(context: dict[str, Any]) -> dict[str, Any]:
    primary = context["assignments"]["primary"]
    sensitivity = context["assignments"]["sensitivity_10s"]
    return {
        "block_s": primary["block_s"],
        "n_blocks": primary["n_blocks"],
        "block_10s_s": sensitivity["block_s"],
        "n_blocks_10s": sensitivity["n_blocks"],
    }


def _dual_level(values: np.ndarray, context: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    primary = context["assignments"]["primary"]
    sensitivity = context["assignments"]["sensitivity_10s"]
    return (
        stats.level(values, primary["blocks"], primary["counts"]),
        stats.level(values, sensitivity["blocks"], sensitivity["counts"]),
    )


def _dual_gap(
    x: np.ndarray, y: np.ndarray, context: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    primary = context["assignments"]["primary"]
    sensitivity = context["assignments"]["sensitivity_10s"]
    return (
        stats.strip(stats.gap(x, y, primary["blocks"], primary["counts"])),
        stats.strip(stats.gap(x, y, sensitivity["blocks"], sensitivity["counts"])),
    )


def _dual_stat_columns(
    prefix: str, primary: dict[str, Any] | None, sensitivity: dict[str, Any] | None,
    include_p: bool = False, include_resolution: bool = False,
) -> dict[str, Any]:
    out = {
        **_stat_columns(prefix, primary, include_p),
        **_stat_columns(f"{prefix}_10s", sensitivity, include_p),
    }
    if include_resolution:
        resolved_primary = bool(
            primary and primary.get("p_raw") is not None and primary["p_raw"] < stats.ALPHA
            and ((primary["ci_low"] is not None and primary["ci_low"] > 0)
                 or (primary["ci_high"] is not None and primary["ci_high"] < 0))
        )
        resolved_10s = bool(
            sensitivity and sensitivity.get("p_raw") is not None
            and sensitivity["p_raw"] < stats.ALPHA
            and ((sensitivity["ci_low"] is not None and sensitivity["ci_low"] > 0)
                 or (sensitivity["ci_high"] is not None and sensitivity["ci_high"] < 0))
        )
        out |= {
            f"{prefix}_resolved_B": resolved_primary,
            f"{prefix}_resolved_10s": resolved_10s,
            f"{prefix}_resolved_both": resolved_primary and resolved_10s,
        }
    return out


def _controls_row(meta: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    c1, c2 = result["controls"]["C-1"], result["controls"]["C-2"]
    c1_10s, c2_10s = result["controls_10s"]["C-1"], result["controls_10s"]["C-2"]
    means = c2["means"]
    means_10s = c2_10s["means"]
    return meta | {
        "eligible": result["eligible"],
        "eligible_10s": result["eligible_10s"],
        "eligible_d10": result["eligible_d10"],
        "eligible_original_5pp": result["eligible_original_5pp"],
        "eligible_original_5pp_10s": result["eligible_original_5pp_10s"],
        "eligible_original_5pp_d10": result["eligible_original_5pp_d10"],
        **_dual_stat_columns("c1", c1, c1_10s, include_p=True, include_resolution=True),
        "c1_pass_original_5pp": c1["pass"],
        "c1_pass_original_5pp_10s": c1_10s["pass"],
        "c1_n_actuating": c1["n_actuating"],
        **_dual_stat_columns("c2", c2, c2_10s, include_p=True, include_resolution=True),
        "c2_mean_fixed_0p1": means[0.1],
        "c2_mean_fixed_1": means[1.0],
        "c2_mean_fixed_5": means[5.0],
        "c2_mean_fixed_0p1_10s": means_10s[0.1],
        "c2_mean_fixed_1_10s": means_10s[1.0],
        "c2_mean_fixed_5_10s": means_10s[5.0],
        "c2_monotone": c2["monotone"],
        "c2_monotone_10s": c2_10s["monotone"],
        "c2_pass": c2["pass"],
        "c2_pass_10s": c2_10s["pass"],
        "tau_s": result["tau_s"],
        "required_block_s": result["block_length_s"],
        "block_pass": result["block_pass"],
        **_block_columns(result),
    }


COMPARATOR = dict(stats.H1_PAIRS)
H1_MODELS = {model for pair in stats.H1_PAIRS for model in pair}


def _l_row(
    meta: dict[str, Any], interpreter: str, by_model: dict[str, dict[str, np.ndarray]],
    context: dict[str, Any],
) -> dict[str, Any]:
    """One L-arm row; the caller guarantees the L arm, its H1 comparator's L arm and the controls are present."""
    arrays = by_model[interpreter]
    aff_level, aff_level_10s = _dual_level(arrays["aff"], context)
    net_level, net_level_10s = _dual_level(arrays["net"], context)
    anchor = COMPARATOR.get(interpreter)
    aff_gap = aff_gap_10s = net_gap = net_gap_10s = None
    if anchor is not None:
        aff_gap, aff_gap_10s = _dual_gap(arrays["aff"], by_model[anchor]["aff"], context)
        net_gap, net_gap_10s = _dual_gap(arrays["net"], by_model[anchor]["net"], context)
    return meta | {
        "interpreter": interpreter,
        "comparator": anchor,
        "contrast_direction": None if anchor is None else f"{interpreter} - {anchor}",
        **_dual_stat_columns("affected", aff_level, aff_level_10s),
        **_dual_stat_columns("network", net_level, net_level_10s),
        **_dual_stat_columns("affected_gap", aff_gap, aff_gap_10s, include_p=True),
        **_dual_stat_columns("network_gap", net_gap, net_gap_10s, include_p=True),
        **_block_columns(context),
    }


HOLM_RULE = "primary_d10"


def _holm_columns(prefix: str, family: str | None, test: dict[str, Any] | None) -> dict[str, Any]:
    if test is None:
        return {f"{prefix}_{key}": None for key in HOLM_KEYS}
    primary, sensitivity = test["primary_B"], test["sensitivity_10s"]
    return {
        f"{prefix}_holm_family": family,
        f"{prefix}_p_holm": primary["p_holm"],
        f"{prefix}_p_holm_10s": sensitivity["p_holm"],
        f"{prefix}_resolved_B": bool(primary["resolved_pos"] or primary["resolved_neg"]),
        f"{prefix}_resolved_10s": bool(sensitivity["resolved_pos"] or sensitivity["resolved_neg"]),
        f"{prefix}_resolved_both": bool(test["resolved_pos"] or test["resolved_neg"]),
    }


def attach_holm_resolution(l_rows: list[dict[str, Any]], rule: dict[str, Any]) -> None:
    """Copy Holm p-values and resolution for L-arm gaps from the hypotheses.json rule result.

    Under ``rule`` (the primary D-10 eligibility rule in production), the base-point affected and
    network gaps are the H1 family, and the affected gap at an eligible RQ2 point is a member of its
    pair's H2 family. Those rows take the Holm-adjusted p-values and both-blocking resolution from
    the very test objects hypotheses.json reports. Every other gap is in no Holm family and its
    Holm and resolution fields stay empty; its raw p-values remain in the ``*_p_raw`` columns.
    """
    members: dict[tuple[str, tuple[str, str], str], tuple[str, dict[str, Any]]] = {}
    h1 = rule["H1"]["result"]
    if h1 is not None:
        for test in h1["tests"]:
            key = (rule["H1"]["base_point"], tuple(test["pair"]), test["metric"])
            members[key] = (f"{HOLM_RULE}:H1", test)
    h2 = rule["H2"]["result"]
    for test in [] if h2 is None else h2["tests"]:
        members[(test["design_point"], tuple(test["pair"]), "aff")] = (f"{HOLM_RULE}:H2", test)
    for row in l_rows:
        pair = (row["interpreter"], row["comparator"])
        for prefix, metric in (("affected_gap", "aff"), ("network_gap", "net")):
            family, test = members.get((row["design_point"], pair, metric), (None, None))
            row.update(_holm_columns(prefix, family, test))


def _an_row(
    meta: dict[str, Any], interpreter: str, l_arrays: dict[str, np.ndarray],
    a_row: dict[str, str], n_row: dict[str, str], located: dict[str, LocatedRun],
    context: dict[str, Any],
) -> dict[str, Any]:
    a = _outcome_arrays(a_row, located, context["stream"])["aff"]
    n = _outcome_arrays(n_row, located, context["stream"])["aff"]
    primary = context["assignments"]["primary"]
    sensitivity = context["assignments"]["sensitivity_10s"]
    decomposition = an_decomposition(
        n, l_arrays["aff"], a, primary["blocks"], primary["counts"])
    decomposition_10s = an_decomposition(
        n, l_arrays["aff"], a, sensitivity["blocks"], sensitivity["counts"])
    return meta | {
        "interpreter": interpreter,
        "outcome": "affected_class_W2",
        "n_minus_l_definition": "accuracy cost at interpreter latency",
        **_dual_stat_columns(
            "n_minus_l", decomposition["n_minus_l"], decomposition_10s["n_minus_l"],
            include_p=True, include_resolution=True),
        "n_minus_a_definition": "latency cost at interpreter accuracy",
        **_dual_stat_columns(
            "n_minus_a", decomposition["n_minus_a"], decomposition_10s["n_minus_a"],
            include_p=True, include_resolution=True),
        **_block_columns(context),
    }


def an_decomposition(
    n: np.ndarray, l: np.ndarray, a: np.ndarray, blocks: np.ndarray, counts: np.ndarray
) -> dict[str, dict[str, Any]]:
    """Affected-class decomposition with the registered signs: N-L and N-A."""
    return {
        "n_minus_l": stats.strip(stats.gap(n, l, blocks, counts)),
        "n_minus_a": stats.strip(stats.gap(n, a, blocks, counts)),
    }


def _mode_rows(
    meta: dict[str, Any], point_rows: list[dict[str, str]], located: dict[str, LocatedRun],
    context: dict[str, Any], interpreter: str,
) -> list[dict[str, Any]]:
    output = []
    for mode in ("E", "P-1", "P-10"):
        row = _one(point_rows, arm="L", mode=mode, interpreter=interpreter)
        arrays = _outcome_arrays(row, located, context["stream"])
        affected, affected_10s = _dual_level(arrays["aff"], context)
        network, network_10s = _dual_level(arrays["net"], context)
        output.append(meta | {
            "interpreter": interpreter,
            "mode": mode,
            **_dual_stat_columns("affected", affected, affected_10s),
            **_dual_stat_columns("network", network, network_10s),
            **_block_columns(context),
        })
    return output


RQ5_ARM = "b-requery-1s"
RQ5_ARMS = ("a", "b", "c")
RQ5_CLASSES = ("video", "xr", "iot", "be")


def _arm_outcomes(run_dir: Path, events: list[dict[str, Any]], stream: dict[str, Any]) -> dict[str, Any]:
    """One trace load: per-intent SLA violation at W = 2 s (as ``_outcome_arrays``), and the run's mean UE
    throughput, mean PRB utilisation and per-class Jain index, all over the slots after T0_S."""
    key = lambda event: (event["j"], event["intent_id"], round(float(event["t_issue_s"]), 6))
    if [key(event) for event in events] != [key(event) for event in stream["events"]]:
        raise RuntimeError(f"{run_dir.name}: events differ from its stream in order, intent or issue time")
    trace = load_run(run_dir)
    slots = SlotTable(trace)
    outcomes = intent_outcomes(trace, events, slots, t_end=trace.sim_time_s)
    for outcome in outcomes:
        if outcome["t_issue_s"] + W_PRIMARY > trace.sim_time_s + 1e-9:
            outcome["aff_W2"] = outcome["net_W2"] = None
    kpi = radio_kpis(trace, T0_S, slots=slots)
    return {
        "aff": stats.as_array(outcomes, "aff_W2"),
        "net": stats.as_array(outcomes, "net_W2"),
        "throughput_mean_mbps": kpi["thr_mean_mbps"],
        "prb_util_mean": kpi["prb_util_mean"],
        "jain": class_jain(slots, T0_S, trace.sim_time_s),
    }


def _rq5_events(
    a_row: dict[str, str], b_row: dict[str, str], oracle_row: dict[str, str], located: dict[str, LocatedRun],
) -> list[dict[str, Any]]:
    """Intent events of RQ5 arm (b). The RQ5 controller runs a (b) row on its arm-(a) row's config, stream and
    N-arm schedule (D-2), so (b) issues the same intents and installs the same policies at the same times as (a);
    only the per-cell class priorities between installs come from the re-queried interpreter instead of the
    numerical xApp. The per-intent windows and the affected class and scope are therefore the N arm's, and the
    affected-class violation is defined for (b) exactly as for (a). The shared inputs are checked on the manifests,
    and the paired b - a and b - c contrasts on the C-4 pairing (UE positions and traffic arrivals byte-identical)."""
    b_dir = located[b_row["run_id"]].run_dir
    a = _manifest(located[a_row["run_id"]].run_dir)
    b = _manifest(b_dir)
    for key in ("schedule_sha256", "stream_sha256", "config_sha256", "sim_time_s", "rng_run"):
        if a.get(key) != b.get(key):
            raise RuntimeError(f"{b_row['run_id']}: {key} differs from its arm (a) {a_row['run_id']}")
    for ref in (a_row, oracle_row):
        same = pairing_identical(b_dir, located[ref["run_id"]].run_dir)
        if not all(same.values()):
            raise RuntimeError(f"{b_row['run_id']}: not paired with {ref['run_id']}: {same}")
    return _events(a_row)


def _rq5_references(
    refs_root: Path, rows: list[dict[str, str]], b_row: dict[str, str], located: dict[str, LocatedRun],
) -> dict[str, LocatedRun]:
    """RQ5 arms (a) and (c) are the M4 reference runs of the base-point N arm and oracle (execution log
    2026-09-27 07:20, D-4): the (b) runs and their references run on the one RQ5 build, so no contrast crosses
    binaries. Each reference must pass verify_run against its matrix row and carry the (b) run's binary_sha256."""
    binary = _manifest(located[b_row["run_id"]].run_dir).get("binary_sha256")
    out = {}
    for row in rows:
        run_dir = refs_root / row["run_id"]
        production = located.get(row["run_id"])
        if production is not None and production.run_dir.resolve() == run_dir.resolve():
            raise RuntimeError(f"RQ5 reference {run_dir} is the production copy of {row['run_id']}")
        state, problems = verify_run(run_dir, row, None)
        if state != "PASS":
            raise RuntimeError(f"RQ5 reference {run_dir}: {state}: {'; '.join(problems)}")
        found = _manifest(run_dir).get("binary_sha256")
        if not binary or found != binary:
            raise RuntimeError(f"RQ5 reference {run_dir}: binary_sha256 {found} differs from (b) "
                               f"{b_row['run_id']} {binary}")
        out[row["run_id"]] = LocatedRun(run_dir, refs_root)
    return out


def _rq5_row(
    meta: dict[str, Any], interpreter: str, run_ids: dict[str, str], arms: dict[str, dict[str, Any]],
    context: dict[str, Any],
) -> dict[str, Any]:
    """RQ5 (exploratory, no Holm): per arm the SLA levels, mean UE throughput and per-class Jain index; paired
    b - a and b - c SLA contrasts on the base point's blockings. Throughput and Jain are per-run aggregates over
    UEs, not per-intent values, so the intent-event block bootstrap gives them no interval."""
    out = meta | {"interpreter": interpreter, **{f"run_id_{arm}": run_ids[arm] for arm in RQ5_ARMS}}
    for arm in RQ5_ARMS:
        for metric, key in (("affected", "aff"), ("network", "net")):
            out |= _dual_stat_columns(f"{metric}_{arm}", *_dual_level(arms[arm][key], context))
        out[f"throughput_mean_mbps_{arm}"] = arms[arm]["throughput_mean_mbps"]
        if set(arms[arm]["jain"]) != set(RQ5_CLASSES):
            raise RuntimeError(f"{run_ids[arm]}: classes {sorted(arms[arm]['jain'])}, expected {RQ5_CLASSES}")
        out |= {f"jain_{cls}_{arm}": arms[arm]["jain"][cls] for cls in RQ5_CLASSES}
    for ref in ("a", "c"):
        for metric, key in (("affected", "aff"), ("network", "net")):
            out |= _dual_stat_columns(
                f"{metric}_b_minus_{ref}", *_dual_gap(arms["b"][key], arms[ref][key], context))
    return out | _block_columns(context)


def _rq3_load(row: dict[str, str]) -> dict[str, Any]:
    """Utilisation of the load trace an RQ3 replay was cut from: the integrity.json that the RQ3 inputs root's
    input-manifest.json pins (sha256-checked) for this interpreter and rate (the scale point uses the 1/s trace)."""
    manifest = json.loads(
        (Path(row["stream_file"]).parents[1] / "input-manifest.json").read_text(encoding="utf-8"))
    entry = manifest["load_traces"][f"{row['interpreter']}@{float(row['rate_per_s']):g}"]
    integrity_path = Path(entry["integrity_path"])
    if sha256_file(integrity_path) != entry["integrity_sha256"]:
        raise RuntimeError(f"{row['run_id']}: {integrity_path} differs from the input manifest")
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    rho, non_stationary = float(integrity["rho"]), bool(integrity["non_stationary"])
    if non_stationary != (rho >= 1.0):
        raise RuntimeError(f"{integrity_path}: non_stationary={non_stationary} but rho={rho}")
    return {"rho": rho, "non_stationary": non_stationary}


def _rq3_row(row: dict[str, str], located: dict[str, LocatedRun]) -> dict[str, Any]:
    """RQ3 radio replay (exploratory). A replay design point has only its N-arm run and no controls, so no B
    (D-10) exists: the SLA intervals use the registered 10 s blocking of the replay stream. A replay whose load
    trace has rho >= 1 is non-stationary (analysis plan): flagged, point estimates only, no block inference.
    Share enforced within 1 s = intents whose policy is installed (actuatable) within 1 s of issue, over all
    replayed intents, the definition of rq3_load.csv on the full load trace."""
    run_dir = located[row["run_id"]].run_dir
    stream = json.loads((run_dir / "stream.json").read_text(encoding="utf-8"))
    events = _events(row)
    arm = _arm_outcomes(run_dir, events, stream)
    load = _rq3_load(row)
    meta = stream["meta"]
    if float(meta["block_s"]) != stats.MIN_BLOCK_S:
        raise RuntimeError(f"{row['run_id']}: stream block {meta['block_s']} s is not the registered 10 s")
    t_issue = np.array([event["t_issue_s"] for event in stream["events"]], dtype=float)
    blocking = stats.block_assignment(t_issue, float(meta["t0_s"]), float(meta["block_s"]), int(meta["n_blocks"]))
    levels = {}
    for metric, key in (("affected", "aff"), ("network", "net")):
        level = stats.level(arm[key], blocking["blocks"], blocking["counts"])
        if load["non_stationary"]:
            level |= {"ci_low": None, "ci_high": None}
        levels |= _stat_columns(metric, level)
    within = [event["install"] is not None and event["e_s"] is not None
              and event["e_s"] - event["t_issue_s"] <= 1.0 for event in events]
    return _point_meta(row["design_point"], [row]) | {
        "interpreter": row["interpreter"],
        "run_id": row["run_id"],
        "rho": load["rho"],
        "non_stationary": load["non_stationary"],
        "block_inference": not load["non_stationary"],
        "block_s": blocking["block_s"],
        "n_blocks": blocking["n_blocks"],
        **levels,
        "prb_util_mean": arm["prb_util_mean"],
        "share_enforced_within_1s": float(np.mean(within)),
        "n_events": len(events),
    }


def _dual_h1(
    arrays: dict[str, dict[str, np.ndarray]], assignments: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    primary_assignment = assignments["primary"]
    sensitivity_assignment = assignments["sensitivity_10s"]
    primary = stats.h1(arrays, primary_assignment["blocks"], primary_assignment["counts"])
    sensitivity = stats.h1(
        arrays, sensitivity_assignment["blocks"], sensitivity_assignment["counts"])
    primary_tests = {(tuple(t["pair"]), t["metric"]): t for t in primary["tests"]}
    sensitivity_tests = {(tuple(t["pair"]), t["metric"]): t for t in sensitivity["tests"]}
    tests = []
    for key in primary_tests:
        pos, neg = stats.resolved_both(primary_tests[key], sensitivity_tests[key], "p_holm")
        tests.append({
            "pair": list(key[0]), "metric": key[1],
            "primary_B": primary_tests[key],
            "sensitivity_10s": sensitivity_tests[key],
            "resolved_pos": pos, "resolved_neg": neg,
        })
    holds = {}
    for left, right in stats.H1_PAIRS:
        pair_tests = [t for t in tests if t["pair"] == [left, right]]
        holds[f"{left} vs {right}"] = (
            len(pair_tests) == 2 and all(t["resolved_pos"] for t in pair_tests)
        ) if pair_tests else None
    return {
        **_assignment_summary(assignments),
        "primary_B": primary,
        "sensitivity_10s": sensitivity,
        "tests": tests,
        "holds": holds,
    }


def _dual_h2(points: dict[str, dict[str, Any]]) -> dict[str, Any]:
    primary_points = {
        name: {
            "eligible": point["eligible"],
            "blocks": point["assignments"]["primary"]["blocks"],
            "counts": point["assignments"]["primary"]["counts"],
            "aff": point["aff"],
        }
        for name, point in points.items()
    }
    sensitivity_points = {
        name: {
            "eligible": point["eligible"],
            "blocks": point["assignments"]["sensitivity_10s"]["blocks"],
            "counts": point["assignments"]["sensitivity_10s"]["counts"],
            "aff": point["aff"],
        }
        for name, point in points.items()
    }
    primary = stats.h2(primary_points)
    sensitivity = stats.h2(sensitivity_points)
    primary_tests = {(t["design_point"], tuple(t["pair"])): t for t in primary["tests"]}
    sensitivity_tests = {(t["design_point"], tuple(t["pair"])): t for t in sensitivity["tests"]}
    tests = []
    for key in primary_tests:
        pos, neg = stats.resolved_both(primary_tests[key], sensitivity_tests[key], "p_holm")
        assignment_summary = _assignment_summary(points[key[0]]["assignments"])
        tests.append({
            "design_point": key[0], "pair": list(key[1]),
            "primary_B": primary_tests[key],
            "sensitivity_10s": sensitivity_tests[key],
            "resolved_pos": pos, "resolved_neg": neg,
            **assignment_summary,
        })
    n_eligible = sum(bool(point["eligible"]) for point in points.values())
    holds = {}
    for left, right in stats.H1_PAIRS:
        pair_tests = [t for t in tests if t["pair"] == [left, right]]
        if not pair_tests:
            holds[f"{left} vs {right}"] = None
            continue
        share = sum(t["resolved_pos"] for t in pair_tests) / len(pair_tests)
        n_reversed = sum(t["resolved_neg"] for t in pair_tests)
        holds[f"{left} vs {right}"] = {
            "share_resolved": share,
            "n_reversed_resolved": n_reversed,
            "n_points": len(pair_tests),
            "holds": bool(
                n_eligible >= stats.H2_MIN_ELIGIBLE
                and share >= stats.H2_SHARE and n_reversed == 0),
        }
    return {
        "n_eligible": n_eligible,
        "block_assignments_by_point": {
            name: _assignment_summary(point["assignments"]) for name, point in points.items()
        },
        "primary_B": primary,
        "sensitivity_10s": sensitivity,
        "tests": tests,
        "holds": holds,
    }


def _assignment_summary(assignments: dict[str, dict[str, Any]]) -> dict[str, Any]:
    primary = assignments["primary"]
    sensitivity = assignments["sensitivity_10s"]
    return {
        "block_s": primary["block_s"], "n_blocks": primary["n_blocks"],
        "block_10s_s": sensitivity["block_s"], "n_blocks_10s": sensitivity["n_blocks"],
    }


def hypotheses(
    point_data: dict[str, dict[str, Any]], pending_points: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Wire H1/H2 under D-10 and the original 5 pp sensitivity eligibility rule.

    ``point_data`` holds the design points whose five controls were analysed; ``point["unavailable"]`` lists the
    absent H1-pair L-arm run ids there. ``pending_points`` holds the core design points whose controls are
    incomplete (``{"is_rq2", "unavailable"}``), so their eligibility is unknown. A Holm family that would need an
    absent run is not computed on the runs at hand: its result is None and ``pending_missing_runs`` names the
    runs it waits for.
    """
    pending_points = pending_points or {}
    rq2_names = sorted(name for name, point in point_data.items() if point["is_rq2"])
    expected_rq2 = len(RQ2_RATES) * len(RQ2_SPEEDS)

    def one_rule(flag: str, label: str) -> dict[str, Any]:
        base = point_data.get(BASE_POINT)
        h1_missing = list(pending_points.get(BASE_POINT, {}).get("unavailable", []))
        if base is not None and base[flag]:
            h1_missing += base.get("unavailable", [])
        h2_missing = [run_id for point in pending_points.values() if point["is_rq2"]
                      for run_id in point["unavailable"]]
        h2_missing += [run_id for point in point_data.values() if point["is_rq2"] and point[flag]
                       for run_id in point.get("unavailable", [])]
        base_eligible = bool(base and base[flag]) and not h1_missing
        h1_result = _dual_h1(base["arrays"], base["assignments"]) if base_eligible else None
        h2_points = {
            name: {
                "eligible": point[flag],
                "assignments": point["assignments"],
                "aff": {model: arrays["aff"] for model, arrays in point["arrays"].items()},
            }
            for name, point in point_data.items() if point["is_rq2"]
        }
        result = {
            "eligibility_rule": label,
            "H1": {
                "tested": base_eligible,
                "base_point": BASE_POINT,
                "base_point_available": base is not None,
                "base_point_eligible": base_eligible,
                "result": h1_result,
            },
            "H2": {
                "complete": len(rq2_names) == expected_rq2,
                "n_prespecified_points": expected_rq2,
                "n_complete_points": len(rq2_names),
                "complete_points": rq2_names,
                "result": None if h2_missing else _dual_h2(h2_points),
            },
        }
        if h1_missing:
            result["H1"]["pending_missing_runs"] = h1_missing
        if h2_missing:
            result["H2"]["pending_missing_runs"] = h2_missing
        return result

    primary_d10 = one_rule(
        "eligible_d10", "D-9 positive floor resolved under both D-10 block analyses")
    return {
        "labels": {
            "H1": "confirmatory test under the outcome-blinded D-9 and D-10 amendments",
            "H2": "confirmatory test under the outcome-blinded D-9 and D-10 amendments",
        },
        "primary_d10": primary_d10,
        "primary_d9": primary_d10,
        "sensitivity_original_5pp": one_rule(
            "eligible_original_5pp_d10",
            "original C-1 >= 5 percentage points rule, resolved under both D-10 block analyses"),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(_jsonable(rows))


META_FIELDS = ["design_point", "rqs", "rate_per_s", "speed_kmh", "ues_per_cell"]
STAT_FIELDS = ["point", "ci_low", "ci_high", "n"]
BLOCK_FIELDS = ["block_s", "n_blocks", "block_10s_s", "n_blocks_10s"]


def _fields(prefix: str, p: bool = False) -> list[str]:
    fields = [f"{prefix}_{key}" for key in STAT_FIELDS]
    return fields + ([f"{prefix}_p_raw"] if p else [])


def _dual_fields(prefix: str, p: bool = False, resolution: bool = False) -> list[str]:
    fields = _fields(prefix, p) + _fields(f"{prefix}_10s", p)
    if resolution:
        fields += [
            f"{prefix}_resolved_B", f"{prefix}_resolved_10s", f"{prefix}_resolved_both",
        ]
    return fields


HOLM_KEYS = (
    "holm_family", "p_holm", "p_holm_10s", "resolved_B", "resolved_10s", "resolved_both",
)


def _holm_fields(prefix: str) -> list[str]:
    return [f"{prefix}_{key}" for key in HOLM_KEYS]


def _radio_and_provenance(
    rows: list[dict[str, str]], located: dict[str, LocatedRun],
    status: dict[str, tuple[str, list[str]]], actuating: dict[str, set[int]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Radio KPIs and provenance per completed run. Stale-policy exposure is filled for L arms and controls at
    design points whose controls were analysed (`actuating`: the D-5 actuating intents per design point)."""
    radio, provenance = [], []
    for row in rows:
        run_id = row["run_id"]
        if status[run_id][0] != "PASS":
            continue
        item = located[run_id]
        manifest = _manifest(item.run_dir)
        stderr = (item.run_dir / "stderr.log").read_text(encoding="utf-8")
        d7_lines = sum(bool(D7_MARKER.search(line)) for line in stderr.splitlines())
        d7b_lines = sum(bool(D7B_MARKER.search(line)) for line in stderr.splitlines())
        lena_commit = str(manifest.get("lena_commit", ""))
        lena_note = manifest.get("lena_note")
        if not lena_note:
            lena_note = lena_commit + (" + D-7 patched branch observed" if d7_lines else "") + (
                " + D-7b patched branch observed" if d7b_lines else "")
        provenance.append({
            "run_id": run_id,
            "design_point": row["design_point"],
            "platform_root": str(item.run_dir.resolve().parent),
            "binary_sha256": manifest.get("binary_sha256"),
            "lena_commit": lena_commit,
            "lena_note": lena_note,
            "d7_patch_lines": d7_lines,
            "d7b_patch_lines": d7b_lines,
        })
        trace = load_run(item.run_dir)
        slots = SlotTable(trace)
        kpi = radio_kpis(trace, T0_S, slots=slots)
        stale = {"mean_ue_s": None, "median_delay_s": None}
        if row["design_point"] in actuating and (row["arm"] == "L" or row["interpreter"] == "control"):
            stale = stale_policy_exposure(trace, _events(row), actuating[row["design_point"]], slots)
        radio.append({
            "run_id": run_id,
            "design_point": row["design_point"],
            "rqs": row["rqs"],
            "mode": row["mode"],
            "arm": row["arm"],
            "interpreter": row["interpreter"],
            "throughput_mean_mbps": kpi["thr_mean_mbps"],
            "throughput_p5_mbps": kpi["thr_p5_mbps"],
            "delay_p50_ms": kpi["delay_p50_ms"],
            "delay_p95_ms": kpi["delay_p95_ms"],
            "delay_p99_ms": kpi["delay_p99_ms"],
            "jitter_iqr_ms": kpi["jitter_iqr_ms"],
            "outage_share": kpi["outage_share"],
            "prb_util_mean": kpi["prb_util_mean"],
            "prb_util_per_cell": json.dumps(kpi["prb_util_per_cell"], sort_keys=True),
            "handover_rate_per_ue_s": kpi["ho_rate_per_ue_s"],
            "handover_total_time_p50_ms": kpi["ho_total_time_p50_ms"],
            "handover_total_time_p95_ms": kpi["ho_total_time_p95_ms"],
            "rlf_count": kpi["rlf_count"],
            "stale_exposure_ue_s_per_intent": stale["mean_ue_s"],
            "stale_delay_median_s": stale["median_delay_s"],
        })
    return radio, provenance


def analyse(matrix: Path, runs_roots: list[Path], out: Path, allow_incomplete: bool = False,
            rq5_refs_root: Path | None = None) -> dict[str, Any]:
    rows = _matrix_rows(matrix)
    located = discover_run_dirs(rows, runs_roots)
    status = _verify(rows, located, allow_incomplete)
    by_point: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_point.setdefault(row["design_point"], []).append(row)

    skipped: list[dict[str, Any]] = []
    controls_csv: list[dict[str, Any]] = []
    l_csv: list[dict[str, Any]] = []
    an_csv: list[dict[str, Any]] = []
    modes_csv: list[dict[str, Any]] = []
    rq5_csv: list[dict[str, Any]] = []
    rq3_csv: list[dict[str, Any]] = []
    point_data: dict[str, dict[str, Any]] = {}
    actuating: dict[str, set[int]] = {}

    pending_points: dict[str, dict[str, Any]] = {}
    core_controls: dict[str, list[dict[str, str]]] = {}

    def absent(required: list[dict[str, str]]) -> list[str]:
        return [row["run_id"] for row in required if status[row["run_id"]][0] != "PASS"]

    # Dependency rule (--allow-incomplete): a unit is computed only when every run it needs is present; otherwise
    # it has no row and skipped.json names it with its absent runs. The five controls fix B (D-10) and C-1, so
    # every unit at a design point needs them; an L-arm row also needs its H1 comparator's L arm.
    core_points = {
        name: point_rows for name, point_rows in by_point.items()
        if any(row["interpreter"] == "control" and row["arm"] in PILOT_ARMS for row in point_rows)
    }
    for design_point, point_rows in sorted(core_points.items()):
        meta = _point_meta(design_point, [row for row in point_rows if row["interpreter"] == "control"])
        control_rows = [_one(point_rows, arm=arm, interpreter="control") for arm in PILOT_ARMS]
        core_controls[design_point] = control_rows
        control_rq = control_rows[0]["rqs"]
        l_by_model = {model: _one(point_rows, arm="L", mode="E", interpreter=model) for model in INTERPRETERS}
        context: dict[str, Any] | None = None
        arrays: dict[str, dict[str, np.ndarray]] = {}
        if _available(control_rows, status, skipped, "controls", design_point):
            context = _control_analysis(design_point, point_rows, located)
            actuating[design_point] = {
                row["j"] for row in context["actuation"]["intents"] if row["actuating"]}
            controls_csv.append(_controls_row(meta, context))
            arrays = {model: _outcome_arrays(row, located, context["stream"])
                      for model, row in l_by_model.items() if not absent([row])}
        else:
            pending_points[design_point] = {"is_rq2": control_rq == "RQ2", "unavailable": absent(control_rows)}

        for interpreter in INTERPRETERS:
            required = control_rows + [l_by_model[interpreter]]
            if interpreter in COMPARATOR:
                required.append(l_by_model[COMPARATOR[interpreter]])
            if _available(required, status, skipped, "L-arm contrast", design_point, interpreter):
                l_csv.append(_l_row(meta, interpreter, arrays, context))
        if context is not None:
            point_data[design_point] = {
                "is_rq2": control_rq == "RQ2",
                "eligible": context["eligible"],
                "eligible_10s": context["eligible_10s"],
                "eligible_d10": context["eligible_d10"],
                "eligible_original_5pp": context["eligible_original_5pp"],
                "eligible_original_5pp_10s": context["eligible_original_5pp_10s"],
                "eligible_original_5pp_d10": context["eligible_original_5pp_d10"],
                "assignments": context["assignments"],
                "arrays": arrays,
                "unavailable": absent([l_by_model[model] for model in INTERPRETERS if model in H1_MODELS]),
            }

        for interpreter in INTERPRETERS:
            candidates = [
                row for row in point_rows
                if row["mode"] == "E" and row["interpreter"] == interpreter and row["arm"] in ("A", "N")
            ]
            if not candidates:
                continue
            a_row = _one(point_rows, arm="A", mode="E", interpreter=interpreter)
            n_row = _one(point_rows, arm="N", mode="E", interpreter=interpreter)
            l_row = _one(point_rows, arm="L", mode="E", interpreter=interpreter)
            required = control_rows + [l_row, a_row, n_row]
            if _available(required, status, skipped, "A/N decomposition", design_point, interpreter):
                an_csv.append(_an_row(
                    meta, interpreter, arrays[interpreter], a_row, n_row, located, context))

        if design_point == BASE_POINT:
            for interpreter in INTERPRETERS:
                mode_rows = [
                    _one(point_rows, arm="L", mode=mode, interpreter=interpreter)
                    for mode in ("E", "P-1", "P-10")
                ]
                if _available(control_rows + mode_rows, status, skipped, "RQ1 modes", design_point, interpreter):
                    modes_csv.extend(_mode_rows(
                        meta, point_rows, located, context, interpreter))

            # RQ5 (exploratory): (a) the mode-E N arm, (b) the re-query run, (c) the oracle control, on one stream.
            oracle_row = _one(point_rows, arm="oracle", interpreter="control")
            oracle_arm: dict[str, Any] | None = None
            for interpreter in INTERPRETERS:
                if not any(row["arm"] == RQ5_ARM and row["interpreter"] == interpreter for row in point_rows):
                    continue
                b_row = _one(point_rows, arm=RQ5_ARM, interpreter=interpreter)
                a_row = _one(point_rows, arm="N", mode="E", interpreter=interpreter)
                for column, want in (("reused_arm_a_run_id", a_row), ("reused_arm_c_run_id", oracle_row)):
                    if b_row.get(column, want["run_id"]) != want["run_id"]:
                        raise ValueError(f"{b_row['run_id']}: {column}={b_row[column]}, expected {want['run_id']}")
                # (a) and (c) come from the RQ5 reference root, so the production copies of those runs are not
                # needed; the base-point controls still fix B.
                if not _available(control_rows + [b_row], status, skipped, "RQ5 division of labour",
                                  design_point, interpreter):
                    continue
                if rq5_refs_root is None:
                    raise RuntimeError("RQ5 (b) runs are present: --rq5-refs-root (the M4 reference runs of arms "
                                       "(a) and (c)) is required")
                refs = located | _rq5_references(rq5_refs_root, [a_row, oracle_row], b_row, located)
                if oracle_arm is None:
                    oracle_arm = _arm_outcomes(
                        refs[oracle_row["run_id"]].run_dir, _events(oracle_row), context["stream"])
                arms = {
                    "a": _arm_outcomes(refs[a_row["run_id"]].run_dir, _events(a_row), context["stream"]),
                    "b": _arm_outcomes(refs[b_row["run_id"]].run_dir, _rq5_events(a_row, b_row, oracle_row, refs),
                                       context["stream"]),
                    "c": oracle_arm,
                }
                run_ids = {"a": a_row["run_id"], "b": b_row["run_id"], "c": oracle_row["run_id"]}
                rq5_csv.append(_rq5_row(meta | {"rqs": b_row["rqs"]}, interpreter, run_ids, arms, context)
                               | {"reference_root": str(rq5_refs_root)})

    # RQ3 radio replays (exploratory): one N-arm run per replay design point; a unit needs only its own run.
    for design_point, point_rows in sorted(by_point.items()):
        for row in point_rows:
            if row["rqs"] != "RQ3":
                continue
            if len(point_rows) != 1:
                raise ValueError(f"{design_point}: an RQ3 replay design point has {len(point_rows)} runs, expected 1")
            if _available([row], status, skipped, "RQ3 radio replay", design_point, row["interpreter"]):
                rq3_csv.append(_rq3_row(row, located))

    hypothesis_json = hypotheses(point_data, pending_points)
    for rule in ("primary_d10", "sensitivity_original_5pp"):
        for family in ("H1", "H2"):
            missing = hypothesis_json[rule][family].get("pending_missing_runs")
            if missing:
                _available([{"run_id": run_id} for run_id in dict.fromkeys(missing)], status, skipped,
                           f"{family} Holm family", BASE_POINT if family == "H1" else None, rule=rule)
    attach_holm_resolution(l_csv, hypothesis_json[HOLM_RULE])
    radio_csv, runs_csv = _radio_and_provenance(rows, located, status, actuating)
    for row in rows:  # a radio row needs its own run; its stale-policy exposure also needs the controls
        extra = {"mode": row["mode"], "arm": row["arm"], "run_id": row["run_id"]}
        if not _available([row], status, skipped, "radio KPIs", row["design_point"], row["interpreter"], **extra):
            continue
        if row["design_point"] in core_controls and (row["arm"] == "L" or row["interpreter"] == "control"):
            _available(core_controls[row["design_point"]], status, skipped, "stale-policy exposure",
                       row["design_point"], row["interpreter"], **extra)
    unavailable_runs = [
        {"run_id": row["run_id"], "design_point": row["design_point"],
         "status": status[row["run_id"]][0], "detail": "; ".join(status[row["run_id"]][1])}
        for row in rows if status[row["run_id"]][0] != "PASS"
    ]

    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "controls.csv", controls_csv, META_FIELDS + [
        "eligible", "eligible_10s", "eligible_d10", "eligible_original_5pp",
        "eligible_original_5pp_10s", "eligible_original_5pp_d10",
        *_dual_fields("c1", True, True),
        "c1_pass_original_5pp", "c1_pass_original_5pp_10s", "c1_n_actuating",
        *_dual_fields("c2", True, True),
        "c2_mean_fixed_0p1", "c2_mean_fixed_1", "c2_mean_fixed_5", "c2_monotone", "c2_pass",
        "c2_mean_fixed_0p1_10s", "c2_mean_fixed_1_10s", "c2_mean_fixed_5_10s",
        "c2_monotone_10s", "c2_pass_10s", "tau_s", "required_block_s", "block_pass",
        *BLOCK_FIELDS,
    ])
    _write_csv(out / "l_arm_contrasts.csv", l_csv, META_FIELDS + [
        "interpreter", "comparator", "contrast_direction",
        *_dual_fields("affected"), *_dual_fields("network"),
        *_dual_fields("affected_gap", True), *_holm_fields("affected_gap"),
        *_dual_fields("network_gap", True), *_holm_fields("network_gap"),
        *BLOCK_FIELDS,
    ])
    _write_csv(out / "an_decomposition.csv", an_csv, META_FIELDS + [
        "interpreter", "outcome", "n_minus_l_definition", *_dual_fields("n_minus_l", True, True),
        "n_minus_a_definition", *_dual_fields("n_minus_a", True, True), *BLOCK_FIELDS,
    ])
    _write_csv(out / "rq1_modes.csv", modes_csv, META_FIELDS + [
        "interpreter", "mode", *_dual_fields("affected"), *_dual_fields("network"), *BLOCK_FIELDS,
    ])
    rq5_fields = META_FIELDS + ["interpreter", *[f"run_id_{arm}" for arm in RQ5_ARMS], "reference_root"]
    for arm in RQ5_ARMS:
        rq5_fields += [*_dual_fields(f"affected_{arm}"), *_dual_fields(f"network_{arm}"),
                       f"throughput_mean_mbps_{arm}", *[f"jain_{cls}_{arm}" for cls in RQ5_CLASSES]]
    for ref in ("a", "c"):
        rq5_fields += [*_dual_fields(f"affected_b_minus_{ref}"), *_dual_fields(f"network_b_minus_{ref}")]
    _write_csv(out / "rq5.csv", rq5_csv, rq5_fields + BLOCK_FIELDS)
    _write_csv(out / "rq3_radio.csv", rq3_csv, META_FIELDS + [
        "interpreter", "run_id", "rho", "non_stationary", "block_inference", "block_s", "n_blocks",
        *_fields("affected"), *_fields("network"), "prb_util_mean", "share_enforced_within_1s", "n_events",
    ])
    _write_csv(out / "radio_kpis.csv", radio_csv, [
        "run_id", "design_point", "rqs", "mode", "arm", "interpreter",
        "throughput_mean_mbps", "throughput_p5_mbps", "delay_p50_ms", "delay_p95_ms", "delay_p99_ms",
        "jitter_iqr_ms", "outage_share", "prb_util_mean", "prb_util_per_cell", "handover_rate_per_ue_s",
        "handover_total_time_p50_ms", "handover_total_time_p95_ms", "rlf_count",
        "stale_exposure_ue_s_per_intent", "stale_delay_median_s",
    ])
    _write_csv(out / "runs.csv", runs_csv, [
        "run_id", "design_point", "platform_root", "binary_sha256", "lena_commit", "lena_note",
        "d7_patch_lines", "d7b_patch_lines",
    ])
    (out / "hypotheses.json").write_text(
        json.dumps(_jsonable(hypothesis_json), indent=2) + "\n", encoding="utf-8")
    skipped_json = {"analysis_units": skipped, "unavailable_runs": unavailable_runs}
    (out / "skipped.json").write_text(
        json.dumps(_jsonable(skipped_json), indent=2) + "\n", encoding="utf-8")
    return {
        "out": str(out),
        "controls": len(controls_csv),
        "l_arm_rows": len(l_csv),
        "an_rows": len(an_csv),
        "rq1_mode_rows": len(modes_csv),
        "rq5_rows": len(rq5_csv),
        "rq3_radio_rows": len(rq3_csv),
        "radio_kpi_rows": len(radio_csv),
        "run_rows": len(runs_csv),
        "skipped_analysis_units": len(skipped),
        "unavailable_runs": len(unavailable_runs),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the production EXP-2026-003 C2 analysis tables")
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--runs-root", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--rq5-refs-root", type=Path,
                        help="M4 reference runs of RQ5 arms (a) and (c), on the (b) runs' binary")
    args = parser.parse_args(argv)
    result = analyse(
        Path(args.matrix), [Path(root) for root in args.runs_root], Path(args.out), args.allow_incomplete,
        args.rq5_refs_root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
