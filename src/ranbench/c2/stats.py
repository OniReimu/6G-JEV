"""Paired time-block bootstrap, controls C-1..C-4, H1/H2 decision rules (EXP-2026-003 analysis plan).

Unit = intent event; its block = the contiguous time block of its issue time. Every arm of a design point uses the
same stream, hence the same block of every event and the same block-draw matrix (seed 20260925, 10,000 resamples).
An arm's statistic is the mean over events with a defined value; a paired gap is the mean of per-event differences
over events defined in both arms. "Resolved" = 95 % CI excludes 0 and Holm-adjusted p (CI inversion) < 0.05.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.edgebench.e2e.analysis import block_draws, ci, pvalue
from src.edgebench.scoring import holm_adjust
from src.ranbench.c2.loader import pairing_identical
from src.ranbench.c2.queueing import utilisation  # noqa: F401  (re-exported)

SEED = 20260925
N_RESAMPLES = 10_000
ALPHA = 0.05
MIN_BLOCK_S = 10.0
C1_MIN_GAP = 0.05
C3_TOL = 0.10
H2_SHARE = 0.80
H2_MIN_ELIGIBLE = 5
ANCHOR_HOSTED = "Jev-1.13.0"
ANCHOR_SELF = "SemIf-Qwen3.5-4B"
H1_PAIRS: list[tuple[str, str]] = [
    ("DeepSeek-V4.1-Flash", ANCHOR_HOSTED),
    ("GLM-5.3-Flash", ANCHOR_HOSTED),
    ("Qwen3.8-Flash", ANCHOR_HOSTED),
    ("Qwen3.5-4B-JSON", ANCHOR_SELF),
]


def block_ids(t_issue: np.ndarray, t0: float, block_s: float, n_blocks: int) -> np.ndarray:
    """Block of each event; the last block absorbs a remainder shorter than one block."""
    b = np.floor((np.asarray(t_issue, float) - t0) / block_s + 1e-9).astype(int)
    if (b < 0).any():
        raise ValueError("event issued before t0")
    return np.minimum(b, n_blocks - 1)


def draws(n_blocks: int) -> np.ndarray:
    return block_draws(n_blocks, N_RESAMPLES, SEED)


def block_assignment(
    t_issue: np.ndarray, t0_s: float, block_s: float, n_blocks: int
) -> dict[str, Any]:
    """One explicit block assignment; every event is kept (``block_ids`` convention)."""
    if not math.isfinite(block_s) or block_s <= 0:
        raise ValueError("block length must be positive")
    if n_blocks < 1:
        raise ValueError("at least one block is required")
    return {
        "block_s": float(block_s),
        "t0_s": float(t0_s),
        "n_blocks": n_blocks,
        "blocks": block_ids(t_issue, t0_s, block_s, n_blocks),
        "counts": draws(n_blocks),
    }


def d10_block_assignments(
    t_issue: np.ndarray, t0_s: float, run_end_s: float,
    autocorr_time_by_arm_s: dict[str, float],
    registered_block_s: float, registered_n_blocks: int,
) -> dict[str, Any]:
    """D-10 primary B=max(10 s, tau) and the registered 10 s analysis as sensitivity.

    Primary: n = floor((run_end - t0) / B) blocks of length B; the last block absorbs the
    remainder shorter than B.  Sensitivity: the frozen stream blocking (registered block
    length and block count), unchanged from the registered analysis.
    """
    if len(autocorr_time_by_arm_s) != 5:
        raise ValueError("autocorrelation times for exactly five control arms are required")
    autocorr_values = [float(value) for value in autocorr_time_by_arm_s.values()]
    if any(not math.isfinite(value) or value <= 0 for value in autocorr_values):
        raise ValueError(f"invalid control-arm autocorrelation times: {autocorr_values}")
    if float(registered_block_s) != MIN_BLOCK_S:
        raise ValueError(f"registered block length {registered_block_s} s is not {MIN_BLOCK_S} s")
    tau_s = max(autocorr_values)
    primary_block_s = max(MIN_BLOCK_S, tau_s)
    span_s = run_end_s - t0_s
    if not math.isfinite(span_s) or span_s <= 0:
        raise ValueError("run end must be finite and later than t0")
    primary_n_blocks = math.floor(span_s / primary_block_s + 1e-9)
    if primary_n_blocks < 1:
        raise ValueError(
            f"run span {span_s:g} s from t0={t0_s:g} to {run_end_s:g} "
            f"contains no complete {primary_block_s:g} s block"
        )
    return {
        "tau_s": tau_s,
        "primary": block_assignment(t_issue, t0_s, primary_block_s, primary_n_blocks),
        "sensitivity_10s": block_assignment(
            t_issue, t0_s, float(registered_block_s), int(registered_n_blocks)),
    }


def d10_block_assignments_from_stream(
    stream: dict[str, Any], run_end_s: float,
    autocorr_time_by_arm_s: dict[str, float],
) -> dict[str, Any]:
    """Read a frozen stream into explicit D-10 assignments without modifying it."""
    t_issue = np.array([event["t_issue_s"] for event in stream["events"]], dtype=float)
    meta = stream["meta"]
    return d10_block_assignments(
        t_issue, float(meta["t0_s"]), run_end_s, autocorr_time_by_arm_s,
        float(meta["block_s"]), int(meta["n_blocks"]))


def as_array(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.array([np.nan if r[key] is None else r[key] for r in rows], dtype=float)


def _boot_mean(v: np.ndarray, blocks: np.ndarray, counts: np.ndarray) -> np.ndarray:
    k = counts.shape[1]
    s = np.bincount(blocks, weights=v, minlength=k)
    c = np.bincount(blocks, minlength=k).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (counts * s).sum(axis=1) / (counts * c).sum(axis=1)


def gap(x: np.ndarray, y: np.ndarray, blocks: np.ndarray, counts: np.ndarray) -> dict[str, Any]:
    """Paired mean difference x - y over events defined in both, with its block-bootstrap distribution."""
    ok = np.isfinite(x) & np.isfinite(y)
    d = (x - y)[ok]
    point = float(d.mean()) if ok.any() else None
    boot = _boot_mean(d, blocks[ok], counts)
    lo, hi = ci(boot, point)
    p = pvalue(boot) if point is not None else None
    return {"point": point, "ci_low": lo, "ci_high": hi, "p_raw": p, "n": int(ok.sum()), "boot": boot}


def level(x: np.ndarray, blocks: np.ndarray, counts: np.ndarray) -> dict[str, Any]:
    ok = np.isfinite(x)
    point = float(x[ok].mean()) if ok.any() else None
    lo, hi = ci(_boot_mean(x[ok], blocks[ok], counts), point)
    return {"point": point, "ci_low": lo, "ci_high": hi, "n": int(ok.sum())}


def _resolved(t: dict[str, Any], p_key: str) -> tuple[bool, bool]:
    p = t.get(p_key)
    if p is None or t["ci_low"] is None:
        return False, False
    return t["ci_low"] > 0 and p < ALPHA, t["ci_high"] < 0 and p < ALPHA


def resolved_positive(t: dict[str, Any], p_key: str) -> bool:
    """Whether a positive effect is resolved by the shared CI and p-value rule."""
    return _resolved(t, p_key)[0]


def resolved_both(
    primary: dict[str, Any], sensitivity_10s: dict[str, Any], p_key: str
) -> tuple[bool, bool]:
    """Positive/negative resolution only when both D-10 block analyses resolve it."""
    primary_pos, primary_neg = _resolved(primary, p_key)
    sensitivity_pos, sensitivity_neg = _resolved(sensitivity_10s, p_key)
    return primary_pos and sensitivity_pos, primary_neg and sensitivity_neg


def apply_holm(tests: list[dict[str, Any]]) -> None:
    """Holm over one family; sets p_holm, resolved_pos, resolved_neg."""
    fam = [t for t in tests if t["p_raw"] is not None]
    for t, p in zip(fam, holm_adjust([t["p_raw"] for t in fam])):
        t["p_holm"], t["family_size"] = p, len(fam)
    for t in tests:
        t.setdefault("p_holm", None)
        t["resolved_pos"], t["resolved_neg"] = _resolved(t, "p_holm")


def strip(t: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in t.items() if k != "boot"}


# ---------------------------------------------------------------- controls

def c1_policy_matters(no_update: np.ndarray, oracle: np.ndarray, blocks: np.ndarray, counts: np.ndarray) -> dict:
    """Network-wide SLA(no-update) - SLA(oracle) >= 5 pp at W = 2 s, and resolved."""
    g = gap(no_update, oracle, blocks, counts)
    pos, _ = _resolved(g, "p_raw")
    return strip(g) | {"pass": bool(pos and g["point"] >= C1_MIN_GAP - 1e-12)}  # tolerance: 5 pp exactly passes


def c2_latency_matters(fixed: dict[float, np.ndarray], blocks: np.ndarray, counts: np.ndarray) -> dict:
    """SLA(d = 5) - SLA(d = 0.1) resolved, and SLA non-decreasing over d in {0.1, 1, 5} (point estimates)."""
    g = gap(fixed[5.0], fixed[0.1], blocks, counts)
    pos, _ = _resolved(g, "p_raw")
    common = np.logical_and.reduce([np.isfinite(fixed[d]) for d in (0.1, 1.0, 5.0)])
    means = {d: float(fixed[d][common].mean()) if common.any() else None for d in (0.1, 1.0, 5.0)}
    mono = None not in means.values() and means[0.1] <= means[1.0] <= means[5.0]
    return strip(g) | {"means": means, "monotone": bool(mono), "pass": bool(pos and mono)}


def d5_controls(no_update: np.ndarray, oracle: np.ndarray, fixed: dict[float, np.ndarray], direction: np.ndarray,
                blocks: np.ndarray, counts: np.ndarray) -> dict[str, dict]:
    """Deviation D-5: C-1 and C-2 on the per-intent affected-class SLA violation (W = 2 s) of the actuating intents
    (direction != 0), each signed by the intended direction; non-actuating intents are undefined. C-1 is then the mean
    of direction * (SLA(no-update) - SLA(oracle)) and C-2 the mean of direction * (SLA(5) - SLA(0.1)), with the
    monotonicity check on the direction-signed means. Same bootstrap, threshold and resolution as the frozen rules."""
    direction = np.asarray(direction, float)
    act = direction != 0
    sgn = lambda v: np.where(act, direction * v, np.nan)
    extra = {"n_actuating": int(act.sum()), "measure": "affected class, actuating intents, signed by direction"}
    c1 = c1_policy_matters(sgn(no_update), sgn(oracle), blocks, counts) | extra
    c2 = c2_latency_matters({d: sgn(v) for d, v in fixed.items()}, blocks, counts) | extra
    return {"C-1": c1, "C-2": c2}


def c3_accounting(d_values: list[float], mean_s: list[float], mean_in_scope_ues: float) -> dict:
    """Mean S_j linear in d with slope within +/-10 % of the mean number of in-scope UEs."""
    slope, intercept = np.polyfit(np.asarray(d_values, float), np.asarray(mean_s, float), 1)
    rel = abs(slope - mean_in_scope_ues) / mean_in_scope_ues if mean_in_scope_ues else math.inf
    return {"slope": float(slope), "intercept": float(intercept), "mean_in_scope_ues": mean_in_scope_ues,
            "rel_err": float(rel), "pass": bool(rel <= C3_TOL)}


def c3_from_traces(tr, stream: dict[str, Any], d_values: tuple[float, ...] = (0.1, 1.0, 5.0),
                   mode: str = "E", d_a1: float = 0.0, cmap: dict[str, str] | None = None) -> dict:
    """C-3 on one run's traces: fixed-latency arms at each d, mean S_j against d; the reference count is each
    intent's time-averaged number of in-scope UEs between its enforcement at the smallest and largest d."""
    from src.ranbench.c2.outcomes import SlotTable, intent_outcomes
    from src.ranbench.c2.schedule import build_arm

    st = SlotTable(tr)
    d_sorted = sorted(d_values)
    arms = [build_arm(stream, "fixed", mode, fixed_d=d, d_a1=d_a1, cmap=cmap) for d in d_sorted]
    mean_s = [float(np.mean([r["S_ue_s"] for r in intent_outcomes(tr, a["events"], st)])) for a in arms]
    lo, hi = arms[0]["events"], arms[-1]["events"]  # smallest and largest d
    n_in = [st.exposure(a["e_s"], b["e_s"], *a["true_install"][:2]) / (b["e_s"] - a["e_s"]) for a, b in zip(lo, hi)]
    return c3_accounting(d_sorted, mean_s, float(np.mean(n_in)))


def c4_pairing(dir_oracle, dir_no_update) -> dict:
    same = pairing_identical(dir_oracle, dir_no_update)
    return {"files": same, "pass": all(same.values())}


# ---------------------------------------------------------------- hypotheses

def h1(arrays: dict[str, dict[str, np.ndarray]], blocks: np.ndarray, counts: np.ndarray) -> dict[str, Any]:
    """arrays[model]["aff"|"net"] = per-event SLA violation at W = 2 s, base point, mode E, L arms."""
    tests = []
    for l, a in H1_PAIRS:
        if l not in arrays or a not in arrays:
            continue
        for metric in ("aff", "net"):
            tests.append({"pair": [l, a], "metric": metric} | gap(arrays[l][metric], arrays[a][metric], blocks, counts))
    apply_holm(tests)
    verdicts = {}
    for l, a in H1_PAIRS:
        ts = [t for t in tests if t["pair"] == [l, a]]
        verdicts[f"{l} vs {a}"] = (len(ts) == 2 and all(t["resolved_pos"] for t in ts)) if ts else None
    return {"tests": [strip(t) for t in tests], "holds": verdicts}


def h2(points: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """points[name] = {"eligible": bool, "blocks", "counts", "aff": {model: per-event array}} (mode E, L arms)."""
    tests = []
    for name, dp in points.items():
        if not dp["eligible"]:
            continue
        for l, a in H1_PAIRS:
            if l in dp["aff"] and a in dp["aff"]:
                tests.append({"design_point": name, "pair": [l, a]}
                             | gap(dp["aff"][l], dp["aff"][a], dp["blocks"], dp["counts"]))
    apply_holm(tests)
    n_elig = sum(bool(dp["eligible"]) for dp in points.values())
    verdicts = {}
    for l, a in H1_PAIRS:
        ts = [t for t in tests if t["pair"] == [l, a]]
        if not ts:
            verdicts[f"{l} vs {a}"] = None
            continue
        share = sum(t["resolved_pos"] for t in ts) / len(ts)
        verdicts[f"{l} vs {a}"] = {
            "share_resolved": share, "n_reversed_resolved": sum(t["resolved_neg"] for t in ts), "n_points": len(ts),
            "holds": bool(n_elig >= H2_MIN_ELIGIBLE and share >= H2_SHARE and not any(t["resolved_neg"] for t in ts)),
        }
    return {"n_eligible": n_elig, "tests": [strip(t) for t in tests], "holds": verdicts}


# ---------------------------------------------------------------- RQ3 and pilot



def autocorr_time_s(series: np.ndarray, dt: float, c: float = 5.0) -> float:
    """Integrated autocorrelation time (Sokal's automatic window, M >= c tau) of a per-slot series, in seconds."""
    x = np.asarray(series, float) - np.mean(series)
    n = len(x)
    if n < 2 or not x.any():
        return dt
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    acf /= acf[0]
    tau = 1.0
    for m in range(1, n):
        tau = 1.0 + 2.0 * acf[1 : m + 1].sum()
        if m >= c * tau:
            break
    return max(tau, 1.0) * dt


def block_length_s(series: np.ndarray, dt: float) -> float:
    return max(MIN_BLOCK_S, autocorr_time_s(series, dt))
