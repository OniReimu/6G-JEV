"""EXP-2026-001 analysis pipeline: statistical testing, comms metrics, energy integration, and hypotheses."""
from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
from io import StringIO
import json
from pathlib import Path
import re
from typing import Any
import warnings

import numpy as np
from scipy.stats import binomtest, chi2, wilcoxon

from src.edgebench.contract import Case
from src.edgebench.scoring import (
    aggregate_metrics,
    aggregate_rq1b_metrics,
    bootstrap_ci,
    _bootstrap_ratio_ci,
    _stratified_bootstrap_indices,
    cochrans_q,
    compute_catalog_metrics,
    compute_growth_ratio,
    compute_ratio_of_ratios,
    exact_mcnemar_test,
    holm_adjust,
    invert_bootstrap_pvalue,
    load_ledger_rows,
    paired_wilcoxon_test,
)

HOSTED_MODELS: list[str] = [
    "Jev-1.13.0",
    "DeepSeek-V4.1-Flash",
    "GLM-5.3-Flash",
    "Qwen3.8-Flash",
]

SELFHOSTED_MODELS: list[str] = [
    "SemIf-Qwen3.5-4B",
    "Laya",
    "Qwen3.5-4B-JSON",
]

THE_SIX_MODELS: list[str] = [
    "Jev-1.13.0",
    "DeepSeek-V4.1-Flash",
    "GLM-5.3-Flash",
    "Qwen3.8-Flash",
    "SemIf-Qwen3.5-4B",
    "Laya",
]

MAIN_MODELS: list[str] = THE_SIX_MODELS

REFERENCE_MODEL: str = "Qwen3.5-4B-JSON"

REFERENCE_MODELS: list[str] = [
    "Qwen3.5-4B-JSON",
    "Rule",
    "MiniLM-Reranker",
    "DistilBERT-Clf-All",
    "DistilBERT-Clf-Frozen",
    "DistilBERT-Clf-Retrained",
]

ALL_EVAL_MODELS: list[str] = THE_SIX_MODELS + [REFERENCE_MODEL]

ALL_TABLE_MODELS: list[str] = THE_SIX_MODELS + [REFERENCE_MODEL] + [
    "Rule",
    "MiniLM-Reranker",
    "DistilBERT-Clf-All",
    "DistilBERT-Clf-Frozen",
    "DistilBERT-Clf-Retrained",
]

RQ4_REFERENCE_MODELS: set[str] = {
    "MiniLM-Reranker",
    "DistilBERT-Clf-All",
    "DistilBERT-Clf-Frozen",
    "DistilBERT-Clf-Retrained",
}


def get_model_role(name: str) -> str:
    """Return the role for a model: 'main', 'reference', or 'control'."""
    if name in THE_SIX_MODELS:
        return "main"
    elif name == "Oracle":
        return "control"
    else:
        return "reference"


EXPECTED_CONDITIONS: dict[str, list[str]] = {
    "RQ1a": ["base", "pad_512", "pad_2048", "pad_8192", "pad_16384"],
    "RQ1b": ["k1", "k2", "k4", "k8"],
    "RQ2": [
        "clean",
        "codeswitch",
        "colloquial",
        "defaultbait",
        "keyvalue",
        "negation",
        "noise",
        "revised",
    ],
    "RQ3": [
        "F4_low",
        "F4_medium",
        "F4_high",
        "F6_low",
        "F6_medium",
        "F6_high",
        "F8_low",
        "F8_medium",
        "F8_high",
    ],
    "RQ4": [
        "K4",
        "K15",
        "K64",
        "K128",
        "K254",
        "churn25",
        "churn50",
    ],
}

EXPECTED_REFERENCE_CELLS: dict[str, list[tuple[str, str]]] = {
    "Rule": [
        (rq, cond)
        for rq in ["RQ1a", "RQ1b", "RQ2", "RQ3"]
        for cond in EXPECTED_CONDITIONS[rq]
    ],
    "MiniLM-Reranker": [
        ("RQ4", cond) for cond in EXPECTED_CONDITIONS["RQ4"]
    ],
    "DistilBERT-Clf-All": [
        ("RQ4", cond) for cond in ["K4", "K15", "K64", "K128", "K254"]
    ],
    "DistilBERT-Clf-Frozen": [
        ("RQ4", cond) for cond in ["churn25", "churn50"]
    ],
    "DistilBERT-Clf-Retrained": [
        ("RQ4", cond) for cond in ["churn25", "churn50"]
    ],
}


def strip_model_suffix(name: str) -> str:
    """Strip '@...' platform suffix from a model name."""
    return name.split("@")[0]


def strip_pad_suffix(case_id: str) -> str:
    """Strip '_pad_<n>' suffix from RQ1a case IDs to align with base cases."""
    return re.sub(r"_pad_\d+$", "", case_id)


def extract_case_index(case_id: str) -> str:
    """Extract 4-digit index from case ID (e.g. '0000' from 'F4_low_test_0000')."""
    m = re.search(r"(\d{4})$", case_id)
    return m.group(1) if m else case_id


# ---------------------------------------------------------------------------
# Addendum A1 & Comms Metrics
# ---------------------------------------------------------------------------

def compute_jitter(latencies: list[float] | np.ndarray) -> dict[str, float]:
    """Compute IQR and standard deviation of latency."""
    arr = np.asarray(latencies, dtype=float)
    if len(arr) == 0:
        return {"iqr": float("nan"), "std": float("nan")}
    q75, q25 = np.percentile(arr, [75, 25])
    iqr = float(q75 - q25)
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return {"iqr": iqr, "std": std}


def compute_deadline_violations(
    latencies: list[float] | np.ndarray,
    thresholds: tuple[float, ...] = (0.05, 0.1, 0.2, 0.5, 1.0),
    seed: int = 20260924,
    n_resamples: int = 10000,
) -> dict[float, dict[str, Any]]:
    """Compute P(latency > tau) with 95% bootstrap CI for specified thresholds tau (in seconds)."""
    arr = np.asarray(latencies, dtype=float)
    n = len(arr)
    out: dict[float, dict[str, Any]] = {}
    if n == 0:
        for tau in thresholds:
            out[tau] = {"p": float("nan"), "ci95": (float("nan"), float("nan"))}
        return out

    for tau in thresholds:
        binary = (arr > tau).astype(float)
        p_val = float(np.mean(binary))
        ci = bootstrap_ci(binary, seed=seed, n_resamples=n_resamples)
        out[tau] = {"p": p_val, "ci95": ci}
    return out


def compute_availability(rows: list[dict[str, Any]]) -> float:
    """Compute availability = 1 - (rows with error_type not null) / n."""
    n = len(rows)
    if n == 0:
        return float("nan")
    err_count = sum(1 for r in rows if r.get("error_type") is not None)
    return float(1.0 - (err_count / n))


def compute_throughput(latencies: list[float] | np.ndarray) -> float:
    """Compute single-client throughput = 1 / mean(latency) (decisions/s)."""
    arr = np.asarray(latencies, dtype=float)
    if len(arr) == 0:
        return float("nan")
    mean_lat = float(np.mean(arr))
    return float(1.0 / mean_lat) if mean_lat > 0 else float("nan")


def compute_goodput(
    em_list: list[bool] | np.ndarray, latencies: list[float] | np.ndarray
) -> float:
    """Compute goodput = EM-correct decisions / sum(latency_s) (decisions/s)."""
    ems = np.asarray(em_list, dtype=bool)
    lats = np.asarray(latencies, dtype=float)
    if len(lats) == 0 or len(ems) == 0:
        return float("nan")
    tot_lat = float(np.sum(lats))
    correct = int(np.sum(ems))
    return float(correct / tot_lat) if tot_lat > 0 else float("nan")


def compute_sla_violation_rate(rows: list[dict[str, Any]]) -> float:
    """Compute SLA-violation rate: share of rows with unsafe_locality or missed_count > 0."""
    n = len(rows)
    if n == 0:
        return float("nan")
    viol_count = sum(
        1
        for r in rows
        if bool(r.get("unsafe_locality", False)) or (int(r.get("missed_count", 0) or 0) > 0)
    )
    return float(viol_count / n)


def compute_false_admission_blocking(
    rows: list[dict[str, Any]],
    cases_map: dict[str, Case] | None = None,
) -> dict[str, float]:
    """Compute RQ4 false-admission and false-blocking rates.

    False admission: truth unsupported, predicted a service (count / truth unsupported count).
    False blocking: truth a service, predicted unsupported (count / truth service count).
    """
    unsupp_total = 0
    service_total = 0
    false_admission = 0
    false_blocking = 0

    for r in rows:
        cid = str(r["case_id"])
        truth = []
        if cases_map and cid in cases_map:
            truth = cases_map[cid].truth
        elif r.get("truth"):
            truth = r["truth"]

        labels = r.get("labels") or []
        for pred_req, truth_req in zip(labels, truth):
            truth_svc = truth_req.get("service_type")
            pred_svc = pred_req.get("service_type")
            if truth_svc == "unsupported":
                unsupp_total += 1
                if pred_svc is not None and pred_svc != "unsupported":
                    false_admission += 1
            elif truth_svc is not None:
                service_total += 1
                if pred_svc == "unsupported":
                    false_blocking += 1

    fa_rate = (
        float(false_admission / unsupp_total)
        if unsupp_total > 0
        else float("nan")
    )
    fb_rate = (
        float(false_blocking / service_total)
        if service_total > 0
        else float("nan")
    )
    return {
        "false_admission_rate": fa_rate,
        "false_blocking_rate": fb_rate,
        "unsupported_total": unsupp_total,
        "service_total": service_total,
        "false_admission_count": false_admission,
        "false_blocking_count": false_blocking,
    }


def compute_payload_overhead(
    rows: list[dict[str, Any]],
    cases_map: dict[str, Case] | None = None,
) -> dict[str, Any]:
    """Compute request bytes and response bytes medians, plus token in/out medians."""
    req_bytes_list: list[int] = []
    resp_bytes_list: list[int] = []
    in_tokens_list: list[int] = []
    out_tokens_list: list[int] = []

    for r in rows:
        cid = str(r["case_id"])
        if cases_map and cid in cases_map:
            text = cases_map[cid].text
            req_bytes_list.append(len(text.encode("utf-8")))

        raw_resp = r.get("raw_response")
        if raw_resp is not None:
            resp_bytes_list.append(len(str(raw_resp).encode("utf-8")))

        if r.get("input_tokens") is not None:
            in_tokens_list.append(int(r["input_tokens"]))
        if r.get("output_tokens") is not None:
            out_tokens_list.append(int(r["output_tokens"]))

    return {
        "request_bytes_median": (
            float(np.median(req_bytes_list)) if req_bytes_list else float("nan")
        ),
        "response_bytes_median": (
            float(np.median(resp_bytes_list)) if resp_bytes_list else float("nan")
        ),
        "input_tokens_median": (
            float(np.median(in_tokens_list)) if in_tokens_list else float("nan")
        ),
        "output_tokens_median": (
            float(np.median(out_tokens_list)) if out_tokens_list else float("nan")
        ),
    }


# ---------------------------------------------------------------------------
# Power & Energy Integration
# ---------------------------------------------------------------------------

def parse_header_tz(header: str) -> timezone:
    """Parse timezone from power CSV header comment (e.g. tz=+1000)."""
    m = re.search(r"tz=([+-]\d{4})", header)
    if m:
        s = m.group(1)
        sign = 1 if s[0] == "+" else -1
        h = int(s[1:3])
        mins = int(s[3:5])
        return timezone(sign * timedelta(hours=h, minutes=mins))
    return timezone.utc


def parse_power_trace(csv_path: str | Path) -> dict[str, Any]:
    """Parse GPU power trace CSV file.

    Returns dict with header info, times (epoch float), powers (W float),
    and raw sample count.
    """
    csv_path = Path(csv_path)
    with open(csv_path, encoding="utf-8") as f:
        header = f.readline().strip()
        tz = parse_header_tz(header)
        times: list[float] = []
        powers: list[float] = []
        for line in f:
            line_str = line.strip()
            if not line_str or line_str.startswith("#"):
                continue
            parts = [p.strip() for p in line_str.split(",")]
            if len(parts) >= 2:
                try:
                    dt = datetime.strptime(
                        parts[0], "%Y/%m/%d %H:%M:%S.%f"
                    ).replace(tzinfo=tz)
                    times.append(dt.timestamp())
                    powers.append(float(parts[1]))
                except ValueError:
                    continue

    return {
        "file": str(csv_path),
        "header": header,
        "tz": tz,
        "times": np.array(times, dtype=float),
        "powers": np.array(powers, dtype=float),
    }


def compute_job_idle_power(
    job_trace: dict[str, Any],
    first_window_start: float,
    window_s: float = 30.0,
) -> float | None:
    """Compute idle power: median of samples in the window_s seconds before first window of job."""
    times = job_trace["times"]
    powers = job_trace["powers"]
    mask = (times >= (first_window_start - window_s)) & (times < first_window_start)
    idle_samples = powers[mask]
    if len(idle_samples) == 0:
        # Fall back to all samples before first_window_start
        mask_prior = times < first_window_start
        idle_samples = powers[mask_prior]
    if len(idle_samples) == 0:
        return None
    return float(np.median(idle_samples))


def integrate_energy(
    times: np.ndarray,
    powers: np.ndarray,
    t_start: float,
    t_end: float,
    n_decisions: int,
    n_correct: int = 0,
    idle_power: float | None = None,
) -> dict[str, Any]:
    """Integrate power over [t_start, t_end] using trapezoid rule.

    Returns dict with total energy (J), J/decision, J/correct decision,
    idle power, J/decision above idle, sample count, and flag_low_samples.
    """
    if len(times) == 0 or len(powers) == 0 or t_end < t_start:
        return {
            "energy_j": float("nan"),
            "energy_j_per_decision": float("nan"),
            "energy_j_per_correct_decision": float("nan"),
            "idle_power_w": idle_power,
            "energy_j_per_decision_above_idle": float("nan"),
            "n_samples": 0,
            "flag_low_samples": True,
        }

    mask = (times >= t_start) & (times <= t_end)
    win_times = times[mask]
    win_powers = powers[mask]
    n_samples = len(win_times)
    flag_low_samples = n_samples < 10

    if n_samples < 2:
        return {
            "energy_j": float("nan"),
            "energy_j_per_decision": float("nan"),
            "energy_j_per_correct_decision": float("nan"),
            "idle_power_w": idle_power,
            "energy_j_per_decision_above_idle": float("nan"),
            "n_samples": n_samples,
            "flag_low_samples": flag_low_samples,
        }

    # np.trapezoid (NumPy >= 2.0) with fallback
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    total_j = float(trapz_fn(win_powers, win_times))

    j_per_dec = float(total_j / n_decisions) if n_decisions > 0 else float("nan")
    j_per_corr = (
        float(total_j / n_correct)
        if n_correct > 0
        else float("nan")
    )

    if idle_power is not None:
        duration_s = float(win_times[-1] - win_times[0])
        idle_j = idle_power * duration_s
        net_j = max(0.0, total_j - idle_j)
        j_per_dec_above_idle = (
            float(net_j / n_decisions) if n_decisions > 0 else float("nan")
        )
    else:
        j_per_dec_above_idle = float("nan")

    return {
        "energy_j": total_j,
        "energy_j_per_decision": j_per_dec,
        "energy_j_per_correct_decision": j_per_corr,
        "idle_power_w": idle_power,
        "energy_j_per_decision_above_idle": j_per_dec_above_idle,
        "n_samples": n_samples,
        "flag_low_samples": flag_low_samples,
    }


# ---------------------------------------------------------------------------
# Enhanced Statistics & Inverted Bootstrap P-Values
# ---------------------------------------------------------------------------

def invert_bootstrap_pvalue_floored(
    bootstrap_samples: list[float] | np.ndarray,
    null_value: float = 0.0,
    n_resamples: int | None = None,
) -> float:
    """Invert bootstrap distribution into a two-sided p-value floored at 1/(B+1)."""
    p_val = invert_bootstrap_pvalue(bootstrap_samples, null_value=null_value)
    boot = np.asarray(bootstrap_samples, dtype=float)
    b = n_resamples if n_resamples is not None else len(boot)
    floor_val = 1.0 / (b + 1)
    return max(p_val, floor_val)


def compute_ratio_of_ratios_with_pvalue(
    lats_m1_level: list[float] | np.ndarray,
    lats_m1_ref: list[float] | np.ndarray,
    lats_m2_level: list[float] | np.ndarray,
    lats_m2_ref: list[float] | np.ndarray,
    seed: int = 20260924,
    n_resamples: int = 10000,
    strata: list[Any] | np.ndarray | None = None,
    null_value: float = 1.0,
) -> tuple[float, tuple[float, float], float, np.ndarray]:
    """Compute ratio-of-ratios G_m1 / G_m2 with case-paired bootstrap CI and inverted p-value against null_value.

    Returns: (point, (ci_low, ci_high), p_value, resample_ratios).
    """
    a1 = np.asarray(lats_m1_level, dtype=float)
    a0 = np.asarray(lats_m1_ref, dtype=float)
    b1 = np.asarray(lats_m2_level, dtype=float)
    b0 = np.asarray(lats_m2_ref, dtype=float)
    n = len(a1)
    if not (len(a0) == n and len(b1) == n and len(b0) == n):
        raise ValueError("All latency arrays must have the same length")
    if strata is not None and len(strata) != n:
        raise ValueError("strata must have one label per case")
    if n == 0:
        return 0.0, (0.0, 0.0), 1.0, np.array([])

    g1_pt, _ = compute_growth_ratio(a1, a0, seed=seed, n_resamples=n_resamples)
    g2_pt, _ = compute_growth_ratio(b1, b0, seed=seed, n_resamples=n_resamples)
    point = g1_pt / g2_pt if g2_pt > 0 else 0.0

    rng = np.random.default_rng(seed)
    if strata is not None:
        indices = _stratified_bootstrap_indices(strata, n_resamples, rng)
    else:
        indices = rng.integers(0, n, size=(n_resamples, n))
    p50_a1 = np.percentile(a1[indices], 50, axis=1)
    p50_a0 = np.percentile(a0[indices], 50, axis=1)
    p50_b1 = np.percentile(b1[indices], 50, axis=1)
    p50_b0 = np.percentile(b0[indices], 50, axis=1)

    g1 = np.where(p50_a0 > 0, p50_a1 / p50_a0, 0.0)
    g2 = np.where(p50_b0 > 0, p50_b1 / p50_b0, 0.0)
    rors = np.where(g2 > 0, g1 / g2, 0.0)

    ci_low = float(np.percentile(rors, 2.5))
    ci_high = float(np.percentile(rors, 97.5))
    ci_low = min(ci_low, point)
    ci_high = max(ci_high, point)

    p_val = invert_bootstrap_pvalue_floored(rors, null_value=null_value, n_resamples=n_resamples)
    return point, (ci_low, ci_high), p_val, rors


def compute_median_difference_with_pvalue(
    diffs: list[float] | np.ndarray,
    seed: int = 20260924,
    n_resamples: int = 10000,
    strata: list[Any] | np.ndarray | None = None,
    null_value: float = 0.0,
) -> tuple[float, tuple[float, float], float, np.ndarray]:
    """Compute bootstrap CI and inverted p-value on median difference."""
    arr = np.asarray(diffs, dtype=float)
    n = len(arr)
    if n == 0:
        return 0.0, (0.0, 0.0), 1.0, np.array([])
    point = float(np.median(arr))

    rng = np.random.default_rng(seed)
    if strata is not None:
        indices = _stratified_bootstrap_indices(strata, n_resamples, rng)
    else:
        indices = rng.integers(0, n, size=(n_resamples, n))
    resamples = np.percentile(arr[indices], 50, axis=1)

    ci_low = float(np.percentile(resamples, 2.5))
    ci_high = float(np.percentile(resamples, 97.5))
    ci_low = min(ci_low, point)
    ci_high = max(ci_high, point)

    p_val = invert_bootstrap_pvalue_floored(resamples, null_value=null_value, n_resamples=n_resamples)
    return point, (ci_low, ci_high), p_val, resamples


def compute_catalog_gap_with_pvalue(
    rows: list[dict[str, Any]],
    cases_map: dict[str, Case] | None = None,
    null_value: float = 0.10,
    seed: int = 20260924,
    n_resamples: int = 10000,
) -> tuple[float | None, tuple[float, float] | None, float, float | None]:
    """Compute seen - unseen gap, CI, CI upper bound, and inverted bootstrap p-value against null_value (0.10)."""
    cat = compute_catalog_metrics(
        rows, cases_map=cases_map, seed=seed, n_resamples=n_resamples
    )
    gap = cat["seen_unseen_gap"]
    ci = cat["seen_unseen_gap_ci95"]
    upper = cat["seen_unseen_gap_ci_upper"]

    # Re-draw the gap replicates to compute p-value
    n = len(rows)
    if n == 0 or gap is None:
        return gap, ci, 1.0, upper

    seen_flag = np.zeros(n, dtype=bool)
    unseen_flag = np.zeros(n, dtype=bool)
    svc_ok = np.zeros(n, dtype=float)

    for i, r in enumerate(rows):
        cid = str(r["case_id"])
        meta = r.get("meta", {})
        if cases_map and cid in cases_map:
            meta = cases_map[cid].meta or meta
        is_seen = meta.get("seen")
        service_correct = r.get("correct", {}).get("service_type")
        truth = cases_map[cid].truth if (cases_map and cid in cases_map) else []
        unsupported_target = any(t.get("service_type") == "unsupported" for t in truth)

        if is_seen is True and service_correct is not None and not unsupported_target:
            seen_flag[i] = True
            svc_ok[i] = float(bool(service_correct))
        elif is_seen is False and service_correct is not None and not unsupported_target:
            unseen_flag[i] = True
            svc_ok[i] = float(bool(service_correct))

    idx = np.random.default_rng(seed).integers(0, n, size=(n_resamples, n))
    seen_cnt = seen_flag[idx].sum(axis=1)
    unseen_cnt = unseen_flag[idx].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        boot_seen = np.where(
            seen_cnt > 0, (svc_ok * seen_flag)[idx].sum(axis=1) / seen_cnt, np.nan
        )
        boot_unseen = np.where(
            unseen_cnt > 0, (svc_ok * unseen_flag)[idx].sum(axis=1) / unseen_cnt, np.nan
        )
    boot_gap = boot_seen - boot_unseen
    finite_gap = boot_gap[np.isfinite(boot_gap)]
    p_val = (
        invert_bootstrap_pvalue_floored(finite_gap, null_value=null_value, n_resamples=n_resamples)
        if len(finite_gap) > 0
        else 1.0
    )

    return gap, ci, p_val, upper


# ---------------------------------------------------------------------------
# Data Loading & Organization
# ---------------------------------------------------------------------------

def load_cases(data_dir: str | Path) -> dict[str, dict[str, dict[str, Case]]]:
    """Load test cases from data_dir/RQ/cond/test.jsonl -> out[rq][cond][case_id] = Case."""
    data_dir = Path(data_dir)
    out: dict[str, dict[str, dict[str, Case]]] = {}
    for rq, conds in EXPECTED_CONDITIONS.items():
        out[rq] = {}
        for cond in conds:
            p = data_dir / rq / cond / "test.jsonl"
            out[rq][cond] = {}
            if p.exists():
                with open(p, encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            c = Case.from_dict(json.loads(line))
                            out[rq][cond][c.case_id] = c
    return out


def load_all_runs(
    runs_dir: str | Path,
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    """Load repeat-0 ledger rows for all (model, rq, condition).

    Returns mapping: (model, rq, condition) -> list of rows.
    """
    runs_dir = Path(runs_dir)
    runs_by_cell: dict[tuple[str, str, str], list[dict[str, Any]]] = {}

    # 1. Hosted ledgers: runs_dir / RQ / cond / ledger.jsonl
    for rq, conds in EXPECTED_CONDITIONS.items():
        for cond in conds:
            p = runs_dir / rq / cond / "ledger.jsonl"
            if p.exists():
                rows = load_ledger_rows(p)
                for r in rows:
                    if r.get("repeat", 0) == 0:
                        m = strip_model_suffix(r["model"])
                        runs_by_cell.setdefault((m, rq, cond), []).append(r)

    # 2. Selfhosted ledgers: runs_dir / _selfhosted / <subfolder> / RQ / cond / ledger.jsonl
    selfhosted_map = {
        "semif": "SemIf-Qwen3.5-4B",
        "laya": "Laya",
        "qwen_json": "Qwen3.5-4B-JSON",
    }
    for sub, expected_m in selfhosted_map.items():
        for rq, conds in EXPECTED_CONDITIONS.items():
            for cond in conds:
                p = runs_dir / "_selfhosted" / sub / rq / cond / "ledger.jsonl"
                if p.exists():
                    rows = load_ledger_rows(p)
                    for r in rows:
                        if r.get("repeat", 0) == 0:
                            m = strip_model_suffix(r["model"])
                            runs_by_cell.setdefault((m, rq, cond), []).append(r)

    # 3. Reference ledgers: runs_dir / _reference / RQ / cond / ledger.jsonl
    for rq, conds in EXPECTED_CONDITIONS.items():
        for cond in conds:
            p = runs_dir / "_reference" / rq / cond / "ledger.jsonl"
            if p.exists():
                rows = load_ledger_rows(p)
                for r in rows:
                    if r.get("repeat", 0) == 0:
                        m = strip_model_suffix(r["model"])
                        runs_by_cell.setdefault((m, rq, cond), []).append(r)

    return runs_by_cell


def load_all_power_traces(
    runs_dir: str | Path,
) -> dict[str, list[dict[str, Any]]]:
    """Load GPU power traces for self-hosted models.

    Returns mapping: model_key -> list of parsed power traces.
    """
    runs_dir = Path(runs_dir)
    power_traces: dict[str, list[dict[str, Any]]] = {}
    selfhosted_map = {
        "SemIf-Qwen3.5-4B": "semif",
        "Laya": "laya",
        "Qwen3.5-4B-JSON": "qwen_json",
    }
    for model_name, sub in selfhosted_map.items():
        sub_dir = runs_dir / "_selfhosted" / sub
        traces: list[dict[str, Any]] = []
        if sub_dir.exists():
            for pf in sorted(sub_dir.glob("power.*.csv")):
                trace = parse_power_trace(pf)
                traces.append(trace)
        power_traces[model_name] = traces
    return power_traces


# ---------------------------------------------------------------------------
# Coverage Checking
# ---------------------------------------------------------------------------

def check_coverage(
    runs_by_cell: dict[tuple[str, str, str], list[dict[str, Any]]],
    models: list[str] | None = None,
    allow_missing: bool = False,
    check_references: bool | None = None,
) -> tuple[dict[tuple[str, str, str], int], list[tuple[str, str, str, int]], str]:
    """Check coverage across expected models and conditions.

    Exits non-zero (or raises ValueError if allow_missing is False) when any cell has n != 300.
    Returns (coverage_dict, missing_cells, formatted_table_string).
    """
    if models is None:
        models = ALL_EVAL_MODELS
        if check_references is None:
            check_references = True
    elif check_references is None:
        check_references = False

    coverage: dict[tuple[str, str, str], int] = {}
    missing_cells: list[tuple[str, str, str], int] = []

    all_conds: list[tuple[str, str]] = []
    for rq, cond_list in EXPECTED_CONDITIONS.items():
        for cond in cond_list:
            all_conds.append((rq, cond))

    for m in models:
        for rq, cond in all_conds:
            cnt = len(runs_by_cell.get((m, rq, cond), []))
            coverage[(m, rq, cond)] = cnt
            if cnt != 300:
                missing_cells.append((m, rq, cond, cnt))

    # Format coverage table
    lines = ["| Condition | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
    for rq, cond in all_conds:
        row = [f"{rq}/{cond}"]
        for m in models:
            cnt = coverage[(m, rq, cond)]
            row.append(str(cnt) if cnt == 300 else f"**{cnt}**")
        lines.append("| " + " | ".join(row) + " |")

    if check_references:
        ref_lines = [
            "",
            "### Reference Interpreters Coverage",
            "",
            "| Reference Model | Condition | Rows | Status |",
            "|---|---|---|---|",
        ]
        for ref_m, expected_cells in EXPECTED_REFERENCE_CELLS.items():
            for rq, cond in expected_cells:
                cnt = len(runs_by_cell.get((ref_m, rq, cond), []))
                coverage[(ref_m, rq, cond)] = cnt
                if cnt != 300:
                    missing_cells.append((ref_m, rq, cond, cnt))
                    status = f"**FAIL ({cnt})**"
                else:
                    status = "PASS"
                ref_lines.append(f"| {ref_m} | {rq}/{cond} | {cnt} | {status} |")
        lines.extend(ref_lines)

    matrix_str = "\n".join(lines)

    if missing_cells:
        msg = f"Coverage check failed: {len(missing_cells)} cells have n != 300!\n"
        for m, rq, cond, cnt in missing_cells:
            msg += f"  - ({m}, {rq}, {cond}): {cnt} rows\n"
        if not allow_missing:
            print("\nCoverage Matrix:\n", flush=True)
            print(matrix_str, flush=True)
            print(flush=True)
            raise ValueError(msg)
        else:
            warnings.warn(msg, stacklevel=2)

    return coverage, missing_cells, matrix_str


# ---------------------------------------------------------------------------
# Output File Generators
# ---------------------------------------------------------------------------

def generate_cells_csv(
    runs_by_cell: dict[tuple[str, str, str], list[dict[str, Any]]],
    cases_by_rq_cond: dict[str, dict[str, dict[str, Case]]],
    models: list[str] | None = None,
    seed: int = 20260924,
) -> tuple[list[dict[str, Any]], str]:
    """Generate cells.csv rows and formatted CSV string."""
    if models is None:
        models = ALL_TABLE_MODELS

    fieldnames = [
        "model",
        "role",
        "rq",
        "condition",
        "n",
        "em",
        "em_ci_low",
        "em_ci_high",
        "macro_field_acc",
        "macro_field_acc_ci_low",
        "macro_field_acc_ci_high",
        "valid_rate",
        "valid_rate_ci_low",
        "valid_rate_ci_high",
        "unsafe_rate",
        "unsafe_rate_ci_low",
        "unsafe_rate_ci_high",
        "spurious_rate",
        "spurious_rate_ci_low",
        "spurious_rate_ci_high",
        "missed_rate",
        "missed_rate_ci_low",
        "missed_rate_ci_high",
        "rq1b_request_em",
        "rq1b_request_em_ci_low",
        "rq1b_request_em_ci_high",
        "rq1b_message_all_correct",
        "seen_top1",
        "seen_top1_ci_low",
        "seen_top1_ci_high",
        "unseen_top1",
        "unseen_top1_ci_low",
        "unseen_top1_ci_high",
        "unsupported_f1",
        "unsupported_f1_ci_low",
        "unsupported_f1_ci_high",
        "latency_p50",
        "latency_p50_ci_low",
        "latency_p50_ci_high",
        "latency_p95",
        "latency_p95_ci_low",
        "latency_p95_ci_high",
        "latency_p99",
        "output_tokens_median",
        "cost_mean_usd_per_call",
        "cost_usd_per_1000_correct",
        "flag_valid_lt_50",
    ]

    out_rows: list[dict[str, Any]] = []

    for rq, conds in EXPECTED_CONDITIONS.items():
        for cond in conds:
            cases_map = cases_by_rq_cond.get(rq, {}).get(cond, {})
            for m in models:
                role = get_model_role(m)
                if role == "control" or m == "Oracle":
                    continue
                cell_rows = runs_by_cell.get((m, rq, cond), [])
                if not cell_rows:
                    continue
                agg = aggregate_metrics(cell_rows, cases_map=cases_map, seed=seed)
                n = agg["n"]

                # Output tokens median
                out_tokens = [
                    int(r["output_tokens"])
                    for r in cell_rows
                    if r.get("output_tokens") is not None
                ]
                out_tok_median = float(np.median(out_tokens)) if out_tokens else ""

                # Hosted cost
                cost_rows = [
                    float(r["cost_usd"])
                    for r in cell_rows
                    if r.get("cost_usd") is not None
                ]
                mean_cost = (
                    float(np.mean(cost_rows)) if cost_rows and m in HOSTED_MODELS else ""
                )
                cost_per_1k = (
                    agg["cost_per_1000_correct"]
                    if (agg.get("cost_per_1000_correct") is not None
                        and not np.isnan(agg["cost_per_1000_correct"])
                        and m in HOSTED_MODELS)
                    else ""
                )

                # RQ1b metrics
                rq1b_req_em = ""
                rq1b_req_em_low = ""
                rq1b_req_em_high = ""
                rq1b_msg_em = ""
                if rq == "RQ1b":
                    rq1b_agg = aggregate_rq1b_metrics(cell_rows, seed=seed)
                    rq1b_req_em = rq1b_agg["request_level_em"]
                    rq1b_req_em_low, rq1b_req_em_high = rq1b_agg["request_level_em_ci95"]
                    rq1b_msg_em = rq1b_agg["message_em"]

                # RQ4 catalog metrics
                seen_top1 = ""
                seen_top1_low = ""
                seen_top1_high = ""
                unseen_top1 = ""
                unseen_top1_low = ""
                unseen_top1_high = ""
                unsupp_f1 = ""
                unsupp_f1_low = ""
                unsupp_f1_high = ""
                if rq == "RQ4":
                    cat = agg.get("catalog", {})
                    if cat.get("seen_accuracy") is not None:
                        seen_top1 = cat["seen_accuracy"]
                        if cat.get("seen_accuracy_ci95"):
                            seen_top1_low, seen_top1_high = cat["seen_accuracy_ci95"]
                    if cat.get("unseen_accuracy") is not None:
                        unseen_top1 = cat["unseen_accuracy"]
                        if cat.get("unseen_accuracy_ci95"):
                            unseen_top1_low, unseen_top1_high = cat["unseen_accuracy_ci95"]
                    if cat.get("unsupported_f1") is not None:
                        unsupp_f1 = cat["unsupported_f1"]
                        if cat.get("unsupported_f1_ci95"):
                            unsupp_f1_low, unsupp_f1_high = cat["unsupported_f1_ci95"]

                is_rq4_ref = m in RQ4_REFERENCE_MODELS

                row: dict[str, Any] = {
                    "model": m,
                    "role": role,
                    "rq": rq,
                    "condition": cond,
                    "n": n,
                    "em": "" if is_rq4_ref else agg["em"],
                    "em_ci_low": "" if is_rq4_ref else agg["em_ci95"][0],
                    "em_ci_high": "" if is_rq4_ref else agg["em_ci95"][1],
                    "macro_field_acc": "" if is_rq4_ref else agg["macro_field_accuracy"],
                    "macro_field_acc_ci_low": "" if is_rq4_ref else agg["macro_field_accuracy_ci95"][0],
                    "macro_field_acc_ci_high": "" if is_rq4_ref else agg["macro_field_accuracy_ci95"][1],
                    "valid_rate": agg["valid_rate"],
                    "valid_rate_ci_low": agg["valid_rate_ci95"][0],
                    "valid_rate_ci_high": agg["valid_rate_ci95"][1],
                    "unsafe_rate": "" if is_rq4_ref else agg["unsafe_rate"],
                    "unsafe_rate_ci_low": "" if is_rq4_ref else agg["unsafe_rate_ci95"][0],
                    "unsafe_rate_ci_high": "" if is_rq4_ref else agg["unsafe_rate_ci95"][1],
                    "spurious_rate": "" if is_rq4_ref else agg["spurious_rate"],
                    "spurious_rate_ci_low": "" if is_rq4_ref else agg["spurious_rate_ci95"][0],
                    "spurious_rate_ci_high": "" if is_rq4_ref else agg["spurious_rate_ci95"][1],
                    "missed_rate": "" if is_rq4_ref else agg["missed_rate"],
                    "missed_rate_ci_low": "" if is_rq4_ref else agg["missed_rate_ci95"][0],
                    "missed_rate_ci_high": "" if is_rq4_ref else agg["missed_rate_ci95"][1],
                    "rq1b_request_em": rq1b_req_em,
                    "rq1b_request_em_ci_low": rq1b_req_em_low,
                    "rq1b_request_em_ci_high": rq1b_req_em_high,
                    "rq1b_message_all_correct": rq1b_msg_em,
                    "seen_top1": seen_top1,
                    "seen_top1_ci_low": seen_top1_low,
                    "seen_top1_ci_high": seen_top1_high,
                    "unseen_top1": unseen_top1,
                    "unseen_top1_ci_low": unseen_top1_low,
                    "unseen_top1_ci_high": unseen_top1_high,
                    "unsupported_f1": unsupp_f1,
                    "unsupported_f1_ci_low": unsupp_f1_low,
                    "unsupported_f1_ci_high": unsupp_f1_high,
                    "latency_p50": agg["latency_p50"],
                    "latency_p50_ci_low": agg["latency_p50_ci95"][0],
                    "latency_p50_ci_high": agg["latency_p50_ci95"][1],
                    "latency_p95": agg["latency_p95"],
                    "latency_p95_ci_low": agg["latency_p95_ci95"][0],
                    "latency_p95_ci_high": agg["latency_p95_ci95"][1],
                    "latency_p99": agg["latency_p99"],
                    "output_tokens_median": out_tok_median,
                    "cost_mean_usd_per_call": mean_cost,
                    "cost_usd_per_1000_correct": cost_per_1k,
                    "flag_valid_lt_50": bool(agg["valid_rate"] < 0.5),
                }
                out_rows.append(row)

    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(out_rows)
    return out_rows, buf.getvalue()


def generate_comms_csv(
    runs_by_cell: dict[tuple[str, str, str], list[dict[str, Any]]],
    cases_by_rq_cond: dict[str, dict[str, dict[str, Case]]],
    power_traces: dict[str, list[dict[str, Any]]] | None = None,
    models: list[str] | None = None,
    seed: int = 20260924,
) -> tuple[list[dict[str, Any]], str]:
    """Generate comms.csv rows and formatted CSV string (Addendum A1 metrics)."""
    if models is None:
        models = ALL_TABLE_MODELS

    fieldnames = [
        "model",
        "role",
        "rq",
        "condition",
        "n",
        "jitter_iqr_s",
        "jitter_std_s",
        "p_gt_0_050",
        "p_gt_0_050_ci_low",
        "p_gt_0_050_ci_high",
        "p_gt_0_100",
        "p_gt_0_100_ci_low",
        "p_gt_0_100_ci_high",
        "p_gt_0_200",
        "p_gt_0_200_ci_low",
        "p_gt_0_200_ci_high",
        "p_gt_0_500",
        "p_gt_0_500_ci_low",
        "p_gt_0_500_ci_high",
        "p_gt_1_000",
        "p_gt_1_000_ci_low",
        "p_gt_1_000_ci_high",
        "availability",
        "throughput_decisions_per_s",
        "goodput_correct_per_s",
        "sla_violation_rate",
        "rq4_false_admission_rate",
        "rq4_false_blocking_rate",
        "request_bytes_median",
        "response_bytes_median",
        "input_tokens_median",
        "output_tokens_median",
        "energy_j_per_decision",
        "energy_j_per_correct_decision",
        "idle_power_w",
        "energy_j_per_decision_above_idle",
        "flag_low_power_samples",
        "flag_valid_lt_50",
    ]

    out_rows: list[dict[str, Any]] = []

    # Pre-compute first window starts and idle powers per job for self-hosted models
    job_idle_powers: dict[str, dict[str, float | None]] = {}
    if power_traces:
        for m_name, traces in power_traces.items():
            job_idle_powers[m_name] = {}
            for trace in traces:
                tmin = trace["times"][0] if len(trace["times"]) else 0.0
                tmax = trace["times"][-1] if len(trace["times"]) else 0.0
                # Find all condition windows falling into this job
                starts = []
                for (cell_m, rq, cond), c_rows in runs_by_cell.items():
                    if cell_m == m_name:
                        sends = [
                            r["t_send_wall"]
                            for r in c_rows
                            if r.get("t_send_wall") is not None
                        ]
                        if sends:
                            s_min = min(sends)
                            if tmin <= s_min <= tmax:
                                starts.append(s_min)
                if starts:
                    first_win = min(starts)
                    idle_p = compute_job_idle_power(trace, first_win, window_s=30.0)
                    job_idle_powers[m_name][trace["file"]] = idle_p
                else:
                    job_idle_powers[m_name][trace["file"]] = None

    for rq, conds in EXPECTED_CONDITIONS.items():
        for cond in conds:
            cases_map = cases_by_rq_cond.get(rq, {}).get(cond, {})
            for m in models:
                role = get_model_role(m)
                if role == "control" or m == "Oracle":
                    continue
                cell_rows = runs_by_cell.get((m, rq, cond), [])
                if not cell_rows:
                    continue
                n = len(cell_rows)
                lats = [
                    float(r["latency_s"])
                    for r in cell_rows
                    if r.get("latency_s") is not None
                ]
                ems = [bool(r.get("em", False)) for r in cell_rows]
                val_list = [
                    bool(r.get("valid", True))
                    for r in cell_rows
                    if r.get("repeat", 0) == 0
                ]
                cell_valid_rate = float(np.mean(val_list)) if val_list else 0.0
                flag_valid_lt_50 = bool(cell_valid_rate < 0.5)

                jitter = compute_jitter(lats)
                deadlines = compute_deadline_violations(
                    lats, thresholds=(0.05, 0.1, 0.2, 0.5, 1.0), seed=seed
                )
                avail = compute_availability(cell_rows)
                tput = compute_throughput(lats)
                is_rq4_ref = m in RQ4_REFERENCE_MODELS
                gput = "" if is_rq4_ref else compute_goodput(ems, lats)
                sla = "" if is_rq4_ref else compute_sla_violation_rate(cell_rows)
                payload = compute_payload_overhead(cell_rows, cases_map=cases_map)

                # RQ4 rates
                fa_rate = ""
                fb_rate = ""
                if rq == "RQ4":
                    rates = compute_false_admission_blocking(
                        cell_rows, cases_map=cases_map
                    )
                    fa_rate = (
                        rates["false_admission_rate"]
                        if not np.isnan(rates["false_admission_rate"])
                        else ""
                    )
                    fb_rate = (
                        rates["false_blocking_rate"]
                        if not np.isnan(rates["false_blocking_rate"])
                        else ""
                    )

                # Energy integration
                energy_per_dec = ""
                energy_per_corr = ""
                idle_power_w = ""
                energy_above_idle = ""
                flag_low_samples = ""

                if power_traces and m in power_traces:
                    sends = [
                        float(r["t_send_wall"])
                        for r in cell_rows
                        if r.get("t_send_wall") is not None
                    ]
                    recvs = [
                        float(r["t_recv_wall"])
                        for r in cell_rows
                        if r.get("t_recv_wall") is not None
                    ]
                    if sends and recvs:
                        t_start = min(sends)
                        t_end = max(recvs)
                        traces = power_traces[m]
                        # Pick matching trace
                        matching_trace = None
                        for tr in traces:
                            if (
                                len(tr["times"])
                                and tr["times"][0] <= t_start <= tr["times"][-1]
                            ):
                                matching_trace = tr
                                break
                        if matching_trace:
                            idle_p = job_idle_powers.get(m, {}).get(
                                matching_trace["file"]
                            )
                            e_res = integrate_energy(
                                matching_trace["times"],
                                matching_trace["powers"],
                                t_start=t_start,
                                t_end=t_end,
                                n_decisions=n,
                                n_correct=sum(ems),
                                idle_power=idle_p,
                            )
                            energy_per_dec = e_res["energy_j_per_decision"]
                            energy_per_corr = (
                                e_res["energy_j_per_correct_decision"]
                                if not np.isnan(e_res["energy_j_per_correct_decision"])
                                else ""
                            )
                            idle_power_w = (
                                e_res["idle_power_w"]
                                if e_res["idle_power_w"] is not None
                                else ""
                            )
                            energy_above_idle = (
                                e_res["energy_j_per_decision_above_idle"]
                                if not np.isnan(
                                    e_res["energy_j_per_decision_above_idle"]
                                )
                                else ""
                            )
                            flag_low_samples = e_res["flag_low_samples"]

                row = {
                    "model": m,
                    "role": role,
                    "rq": rq,
                    "condition": cond,
                    "n": n,
                    "jitter_iqr_s": jitter["iqr"],
                    "jitter_std_s": jitter["std"],
                    "p_gt_0_050": deadlines[0.05]["p"],
                    "p_gt_0_050_ci_low": deadlines[0.05]["ci95"][0],
                    "p_gt_0_050_ci_high": deadlines[0.05]["ci95"][1],
                    "p_gt_0_100": deadlines[0.1]["p"],
                    "p_gt_0_100_ci_low": deadlines[0.1]["ci95"][0],
                    "p_gt_0_100_ci_high": deadlines[0.1]["ci95"][1],
                    "p_gt_0_200": deadlines[0.2]["p"],
                    "p_gt_0_200_ci_low": deadlines[0.2]["ci95"][0],
                    "p_gt_0_200_ci_high": deadlines[0.2]["ci95"][1],
                    "p_gt_0_500": deadlines[0.5]["p"],
                    "p_gt_0_500_ci_low": deadlines[0.5]["ci95"][0],
                    "p_gt_0_500_ci_high": deadlines[0.5]["ci95"][1],
                    "p_gt_1_000": deadlines[1.0]["p"],
                    "p_gt_1_000_ci_low": deadlines[1.0]["ci95"][0],
                    "p_gt_1_000_ci_high": deadlines[1.0]["ci95"][1],
                    "availability": avail,
                    "throughput_decisions_per_s": tput,
                    "goodput_correct_per_s": gput,
                    "sla_violation_rate": sla,
                    "rq4_false_admission_rate": fa_rate,
                    "rq4_false_blocking_rate": fb_rate,
                    "request_bytes_median": payload["request_bytes_median"],
                    "response_bytes_median": payload["response_bytes_median"],
                    "input_tokens_median": payload["input_tokens_median"],
                    "output_tokens_median": payload["output_tokens_median"],
                    "energy_j_per_decision": energy_per_dec,
                    "energy_j_per_correct_decision": energy_per_corr,
                    "idle_power_w": idle_power_w,
                    "energy_j_per_decision_above_idle": energy_above_idle,
                    "flag_low_power_samples": flag_low_samples,
                    "flag_valid_lt_50": flag_valid_lt_50,
                }
                out_rows.append(row)

    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(out_rows)
    return out_rows, buf.getvalue()


def generate_contrasts_csv(
    runs_by_cell: dict[tuple[str, str, str], list[dict[str, Any]]],
    the_five: list[str] | None = None,
    reference_model: str = "Jev-1.13.0",
    seed: int = 20260924,
) -> tuple[list[dict[str, Any]], str]:
    """Generate contrasts.csv: each of the five vs Jev-1.13.0 per condition, plus Cochran's Q across the six."""
    if the_five is None:
        the_five = [m for m in THE_SIX_MODELS if m != reference_model]

    fieldnames = [
        "rq",
        "condition",
        "model",
        "reference_model",
        "n_paired",
        "em_diff",
        "em_diff_ci_low",
        "em_diff_ci_high",
        "em_mcnemar_p",
        "unsafe_diff",
        "unsafe_diff_ci_low",
        "unsafe_diff_ci_high",
        "unsafe_mcnemar_p",
        "cochrans_q_em_stat",
        "cochrans_q_em_p",
        "cochrans_q_unsafe_stat",
        "cochrans_q_unsafe_p",
    ]

    out_rows: list[dict[str, Any]] = []

    for rq, conds in EXPECTED_CONDITIONS.items():
        for cond in conds:
            jev_rows = runs_by_cell.get((reference_model, rq, cond), [])
            jev_by_cid = {str(r["case_id"]): r for r in jev_rows}

            # Cochran's Q on EM and unsafe across the six models
            cids_six = set(jev_by_cid.keys())
            by_m: dict[str, dict[str, dict[str, Any]]] = {reference_model: jev_by_cid}
            for m in the_five:
                m_rows = runs_by_cell.get((m, rq, cond), [])
                m_by_cid = {str(r["case_id"]): r for r in m_rows}
                by_m[m] = m_by_cid
                cids_six &= set(m_by_cid.keys())
            sorted_cids_six = sorted(cids_six)

            q_em_stat = float("nan")
            q_em_p = float("nan")
            q_un_stat = float("nan")
            q_un_p = float("nan")

            if sorted_cids_six:
                em_mat = [
                    [int(bool(by_m[m][cid].get("em", False))) for m in THE_SIX_MODELS]
                    for cid in sorted_cids_six
                ]
                res_em = cochrans_q(em_mat)
                q_em_stat, q_em_p = res_em["q"], res_em["p"]

                un_mat = [
                    [
                        int(bool(by_m[m][cid].get("unsafe_locality", False)))
                        for m in THE_SIX_MODELS
                    ]
                    for cid in sorted_cids_six
                ]
                res_un = cochrans_q(un_mat)
                q_un_stat, q_un_p = res_un["q"], res_un["p"]

            # Contrasts for each of the five vs Jev-1.13.0
            for m in the_five:
                m_by_cid = by_m.get(m, {})
                common_cids = sorted(set(jev_by_cid.keys()) & set(m_by_cid.keys()))
                n_paired = len(common_cids)
                if n_paired == 0:
                    continue

                m_em = [bool(m_by_cid[c].get("em", False)) for c in common_cids]
                j_em = [bool(jev_by_cid[c].get("em", False)) for c in common_cids]
                diff_em = np.asarray(m_em, dtype=float) - np.asarray(j_em, dtype=float)
                em_diff = float(np.mean(diff_em))
                em_ci = bootstrap_ci(diff_em, seed=seed, n_resamples=10000)
                em_mcnemar = exact_mcnemar_test(m_em, j_em)

                m_un = [bool(m_by_cid[c].get("unsafe_locality", False)) for c in common_cids]
                j_un = [bool(jev_by_cid[c].get("unsafe_locality", False)) for c in common_cids]
                diff_un = np.asarray(m_un, dtype=float) - np.asarray(j_un, dtype=float)
                un_diff = float(np.mean(diff_un))
                un_ci = bootstrap_ci(diff_un, seed=seed, n_resamples=10000)
                un_mcnemar = exact_mcnemar_test(m_un, j_un)

                out_rows.append(
                    {
                        "rq": rq,
                        "condition": cond,
                        "model": m,
                        "reference_model": reference_model,
                        "n_paired": n_paired,
                        "em_diff": em_diff,
                        "em_diff_ci_low": em_ci[0],
                        "em_diff_ci_high": em_ci[1],
                        "em_mcnemar_p": em_mcnemar,
                        "unsafe_diff": un_diff,
                        "unsafe_diff_ci_low": un_ci[0],
                        "unsafe_diff_ci_high": un_ci[1],
                        "unsafe_mcnemar_p": un_mcnemar,
                        "cochrans_q_em_stat": q_em_stat,
                        "cochrans_q_em_p": q_em_p,
                        "cochrans_q_unsafe_stat": q_un_stat,
                        "cochrans_q_unsafe_p": q_un_p,
                    }
                )

    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(out_rows)
    return out_rows, buf.getvalue()


# ---------------------------------------------------------------------------
# Hypotheses H1–H5 Engine
# ---------------------------------------------------------------------------

def generate_flagged_cells_md(
    runs_by_cell: dict[tuple[str, str, str], list[dict[str, Any]]],
    models: list[str] | None = None,
) -> str:
    """Generate 'Flagged cells' markdown section for cells with valid_rate < 0.5."""
    if models is None:
        models = ALL_EVAL_MODELS

    flagged: list[dict[str, Any]] = []
    for rq, conds in EXPECTED_CONDITIONS.items():
        for cond in conds:
            for m in models:
                cell_rows = [
                    r
                    for r in runs_by_cell.get((m, rq, cond), [])
                    if r.get("repeat", 0) == 0
                ]
                if not cell_rows:
                    continue
                v_list = [bool(r.get("valid", True)) for r in cell_rows]
                vr = float(np.mean(v_list)) if v_list else 0.0
                if vr < 0.5:
                    errors = Counter(
                        r.get("error_type")
                        for r in cell_rows
                        if r.get("error_type") is not None
                    )
                    dominant_err = (
                        errors.most_common(1)[0][0] if errors else "None"
                    )
                    raw_ex = ""
                    for r in cell_rows:
                        if r.get("raw_response"):
                            raw_str = r["raw_response"]
                            try:
                                d = (
                                    json.loads(raw_str)
                                    if isinstance(raw_str, str)
                                    else raw_str
                                )
                                raw_ex = (
                                    d.get("error", raw_str)
                                    if isinstance(d, dict)
                                    else str(raw_str)
                                )
                            except Exception:
                                raw_ex = str(raw_str)
                            break
                    flagged.append(
                        {
                            "model": m,
                            "rq": rq,
                            "condition": cond,
                            "valid_rate": vr,
                            "dominant_error": dominant_err,
                            "example_msg": raw_ex,
                        }
                    )

    lines = [
        "## Flagged cells",
        "",
        "Per analysis plan (\"Cells with valid rate < 50% are still tested (no screening), and flagged\"): ",
        "",
    ]
    if flagged:
        for f in flagged:
            lines.append(
                f"- **{f['model']}** ({f['rq']}/{f['condition']}): valid rate = {f['valid_rate']:.1%}, dominant error_type = {f['dominant_error']}, example raw_response: \"{f['example_msg']}\""
            )
    else:
        lines.append("- No cells were flagged with valid rate < 50%.")
    lines.append("")
    return "\n".join(lines)


def generate_h5_classifier_reference(
    runs_by_cell: dict[tuple[str, str, str], list[dict[str, Any]]],
    cases_by_rq_cond: dict[str, dict[str, dict[str, Case]]],
    runs_dir: str | Path | None = None,
    seed: int = 20260924,
) -> tuple[list[dict[str, Any]], str]:
    """Generate descriptive table for frozen and retrained classifier references at churn25 and churn50."""
    fieldnames = [
        "churn_level",
        "condition",
        "model",
        "seen_top1",
        "seen_top1_ci_low",
        "seen_top1_ci_high",
        "unseen_top1",
        "unseen_top1_ci_low",
        "unseen_top1_ci_high",
        "gap",
        "gap_ci_low",
        "gap_ci_high",
        "added_labelled_examples",
        "training_wall_time_s",
    ]

    out_rows: list[dict[str, Any]] = []

    for cond in ["churn25", "churn50"]:
        cases_map = cases_by_rq_cond.get("RQ4", {}).get(cond, {})
        for m in ["DistilBERT-Clf-Frozen", "DistilBERT-Clf-Retrained"]:
            cell_rows = runs_by_cell.get((m, "RQ4", cond), [])
            if not cell_rows:
                continue
            cat = compute_catalog_metrics(cell_rows, cases_map=cases_map, seed=seed)

            # Adaptation cost from training.json
            added_ex: int | str = ""
            wall_time: float | str = ""
            if m == "DistilBERT-Clf-Frozen":
                added_ex = 0
                wall_time = ""
            else:
                train_dir_name = "clf_retrained_25" if cond == "churn25" else "clf_retrained_50"
                cand_dirs: list[Path] = []
                if runs_dir is not None:
                    cand_dirs.append(Path(runs_dir) / "_reference" / "_models" / train_dir_name)
                cand_dirs.append(Path("runs/EXP-2026-001/_reference/_models") / train_dir_name)
                for td in cand_dirs:
                    tp = td / "training.json"
                    if tp.exists():
                        try:
                            with open(tp, encoding="utf-8") as f:
                                tdata = json.load(f)
                            added_ex = tdata.get(
                                "adaptation_examples_count",
                                tdata.get("data_counts", {}).get("adaptation_examples", 0),
                            )
                            wall_time = float(tdata.get("wall_time_s", 0.0))
                            break
                        except Exception:
                            pass

            seen_ci = cat.get("seen_accuracy_ci95") or (float("nan"), float("nan"))
            unseen_ci = cat.get("unseen_accuracy_ci95") or (float("nan"), float("nan"))
            gap_ci = cat.get("seen_unseen_gap_ci95") or (float("nan"), float("nan"))

            row = {
                "churn_level": cond,
                "condition": cond,
                "model": m,
                "seen_top1": cat.get("seen_accuracy", float("nan")),
                "seen_top1_ci_low": seen_ci[0],
                "seen_top1_ci_high": seen_ci[1],
                "unseen_top1": cat.get("unseen_accuracy", float("nan")),
                "unseen_top1_ci_low": unseen_ci[0],
                "unseen_top1_ci_high": unseen_ci[1],
                "gap": cat.get("seen_unseen_gap", float("nan")),
                "gap_ci_low": gap_ci[0],
                "gap_ci_high": gap_ci[1],
                "added_labelled_examples": added_ex,
                "training_wall_time_s": wall_time,
            }
            out_rows.append(row)

    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(out_rows)
    return out_rows, buf.getvalue()


def generate_h5_classifier_reference_md(clf_rows: list[dict[str, Any]]) -> str:
    """Format markdown section for classifier references."""
    lines = [
        "## H5 Classifier References (Descriptive)",
        "",
        "Descriptive comparison against supervised classifier references (DistilBERT):",
        "- **Frozen classifier floor**: Head trained on C_64 v0; unseen accuracy is 0.0% by construction (floor).",
        "- **Retrained classifier**: Head retrained with new churn services added to the label space. Adaptation cost shows added labelled examples and training wall time.",
        "",
        "| Churn Level | Classifier | Seen Top-1 (95% CI) | Unseen Top-1 (95% CI) | Gap (95% CI) | Added Labelled Examples | Training Wall Time (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    if clf_rows:
        for r in clf_rows:
            s_ci = (
                f"[{r['seen_top1_ci_low']:.4f}, {r['seen_top1_ci_high']:.4f}]"
                if isinstance(r.get("seen_top1_ci_low"), (int, float)) and not np.isnan(r["seen_top1_ci_low"])
                else "NA"
            )
            u_ci = (
                f"[{r['unseen_top1_ci_low']:.4f}, {r['unseen_top1_ci_high']:.4f}]"
                if isinstance(r.get("unseen_top1_ci_low"), (int, float)) and not np.isnan(r["unseen_top1_ci_low"])
                else "NA"
            )
            g_ci = (
                f"[{r['gap_ci_low']:.4f}, {r['gap_ci_high']:.4f}]"
                if isinstance(r.get("gap_ci_low"), (int, float)) and not np.isnan(r["gap_ci_low"])
                else "NA"
            )

            s_val = (
                f"{r['seen_top1']:.4f} {s_ci}"
                if isinstance(r.get("seen_top1"), (int, float)) and not np.isnan(r["seen_top1"])
                else "NA"
            )
            u_val = (
                f"{r['unseen_top1']:.4f} {u_ci}"
                if isinstance(r.get("unseen_top1"), (int, float)) and not np.isnan(r["unseen_top1"])
                else "NA"
            )
            g_val = (
                f"{r['gap']:.4f} {g_ci}"
                if isinstance(r.get("gap"), (int, float)) and not np.isnan(r["gap"])
                else "NA"
            )

            ex_str = str(r["added_labelled_examples"]) if r.get("added_labelled_examples") != "" else "—"
            wt_str = (
                f"{r['training_wall_time_s']:.2f}"
                if isinstance(r.get("training_wall_time_s"), (int, float))
                else "—"
            )

            lines.append(
                f"| {r['condition']} | {r['model']} | {s_val} | {u_val} | {g_val} | {ex_str} | {wt_str} |"
            )
    else:
        lines.append("- No classifier reference runs found in the current dataset.")
    lines.append("")
    return "\n".join(lines)


def evaluate_hypotheses(
    runs_by_cell: dict[tuple[str, str, str], list[dict[str, Any]]],
    cases_by_rq_cond: dict[str, dict[str, dict[str, Case]]],
    runs_dir: str | Path | None = None,
    seed: int = 20260924,
) -> tuple[list[dict[str, Any]], str]:
    """Evaluate pre-registered confirmatory hypotheses H1–H5.

    Returns (hypotheses_rows, hypotheses_markdown).
    """
    hypotheses_rows: list[dict[str, Any]] = []

    # Pre-calculate cell valid rates for low-validity flagging (< 50%)
    cell_valid_rates: dict[tuple[str, str], float] = {}
    for (m, rq, cond), c_rows in runs_by_cell.items():
        r0 = [r for r in c_rows if r.get("repeat", 0) == 0]
        vr = float(np.mean([bool(r.get("valid", True)) for r in r0])) if r0 else 0.0
        cell_valid_rates[(m, cond)] = vr

    def _get_flag_low_valid(cells: list[tuple[str, str]]) -> str:
        low = [f"({m}, {c})" for m, c in cells if cell_valid_rates.get((m, c), 1.0) < 0.5]
        return ", ".join(low)

    # Helper to resolve hypothesis decision rule
    def _is_resolved(
        ci_low: float,
        ci_high: float,
        holm_p: float,
        null_val: float,
        direction: str,
    ) -> bool:
        if holm_p >= 0.05:
            return False
        if direction == ">":
            return ci_low > null_val
        elif direction == "<":
            return ci_high < null_val
        elif direction == "!=":
            return ci_low > null_val or ci_high < null_val
        elif direction == "upper_bound_lt":
            # For H5: 95% CI upper bound < 10 pp (0.10)
            return ci_high < null_val
        return False

    # -----------------------------------------------------------------------
    # H1 (RQ1a): Length scalability
    # -----------------------------------------------------------------------
    # H1(a): Hosted LLM - Jev paired latency difference > 0 at valid levels (>= 0.90)
    # Self-hosted analogue: Qwen3.5-4B-JSON - SemIf-Qwen3.5-4B
    h1a_raw_contrasts: list[dict[str, Any]] = []
    gated_out_h1a: list[str] = []

    # Hosted contrasts
    hosted_pairs = [
        (m, "Jev-1.13.0") for m in ["DeepSeek-V4.1-Flash", "GLM-5.3-Flash", "Qwen3.8-Flash"]
    ]
    # Self-hosted analogue
    sh_pairs = [("Qwen3.5-4B-JSON", "SemIf-Qwen3.5-4B")]
    all_h1a_pairs = hosted_pairs + sh_pairs

    for m_cand, m_ref in all_h1a_pairs:
        for lvl in EXPECTED_CONDITIONS["RQ1a"]:
            cand_rows = runs_by_cell.get((m_cand, "RQ1a", lvl), [])
            ref_rows = runs_by_cell.get((m_ref, "RQ1a", lvl), [])
            cand_valid = (
                float(np.mean([bool(r.get("valid", True)) for r in cand_rows]))
                if cand_rows
                else 0.0
            )
            ref_valid = (
                float(np.mean([bool(r.get("valid", True)) for r in ref_rows]))
                if ref_rows
                else 0.0
            )

            contrast_id = f"{m_cand} - {m_ref} @ {lvl}"
            cells_h1a = [(m_cand, lvl), (m_ref, lvl)]
            flag_low_h1a = _get_flag_low_valid(cells_h1a)
            if cand_valid < 0.90 or ref_valid < 0.90:
                gated_out_h1a.append(
                    f"{contrast_id} (valid: {m_cand}={cand_valid:.1%}, {m_ref}={ref_valid:.1%})"
                )
                h1a_raw_contrasts.append(
                    {
                        "hypothesis": "H1(a)",
                        "rq": "RQ1a",
                        "family": "latency_contrast",
                        "contrast": contrast_id,
                        "model": m_cand,
                        "reference_model": m_ref,
                        "level": lvl,
                        "null_val": 0.0,
                        "direction": ">",
                        "estimate": float("nan"),
                        "ci_low": float("nan"),
                        "ci_high": float("nan"),
                        "raw_p": float("nan"),
                        "p_floor": False,
                        "gated": True,
                        "flag_low_valid": flag_low_h1a,
                        "notes": f"Gated out: valid rates {cand_valid:.1%} / {ref_valid:.1%} < 90%",
                    }
                )
                continue

            # Align on case_id (with pad suffix stripped)
            cand_by_id = {
                strip_pad_suffix(r["case_id"]): r
                for r in cand_rows
                if r.get("latency_s") is not None
            }
            ref_by_id = {
                strip_pad_suffix(r["case_id"]): r
                for r in ref_rows
                if r.get("latency_s") is not None
            }
            common = sorted(set(cand_by_id.keys()) & set(ref_by_id.keys()))
            cand_lats = [cand_by_id[c]["latency_s"] for c in common]
            ref_lats = [ref_by_id[c]["latency_s"] for c in common]

            diffs = np.asarray(cand_lats, dtype=float) - np.asarray(ref_lats, dtype=float)
            pt = float(np.median(diffs))
            ci = bootstrap_ci(
                diffs,
                seed=seed,
                n_resamples=10000,
                stat_fn=lambda a, **kw: np.median(a, **kw),
            )
            raw_p = paired_wilcoxon_test(cand_lats, ref_lats)

            h1a_raw_contrasts.append(
                {
                    "hypothesis": "H1(a)",
                    "rq": "RQ1a",
                    "family": "latency_contrast",
                    "contrast": contrast_id,
                    "model": m_cand,
                    "reference_model": m_ref,
                    "level": lvl,
                    "null_val": 0.0,
                    "direction": ">",
                    "estimate": pt,
                    "ci_low": ci[0],
                    "ci_high": ci[1],
                    "raw_p": raw_p,
                    "p_floor": False,
                    "gated": False,
                    "flag_low_valid": flag_low_h1a,
                    "notes": f"n_paired={len(common)}",
                }
            )

    # Holm adjustment for H1(a) within RQ1a latency contrast family
    h1a_valid = [c for c in h1a_raw_contrasts if not c["gated"]]
    h1a_pvals = [c["raw_p"] for c in h1a_valid]
    h1a_adj = holm_adjust(h1a_pvals)
    for c, adj_p in zip(h1a_valid, h1a_adj):
        c["holm_p"] = adj_p
        c["verdict"] = (
            "resolved"
            if _is_resolved(
                c["ci_low"], c["ci_high"], adj_p, c["null_val"], c["direction"]
            )
            else "not resolved"
        )
    for c in h1a_raw_contrasts:
        if c["gated"]:
            c["holm_p"] = float("nan")
            c["verdict"] = "gated out"

    # H1(b): Growth ratio G_Jev(16K)/G_l(16K) < 1.0 (ratio-of-ratios)
    # Self-hosted analogue: SemIf vs Qwen-JSON
    h1b_raw_contrasts: list[dict[str, Any]] = []

    def _eval_h1b(m1: str, m2: str, label_cand: str, label_ref: str) -> dict[str, Any]:
        m1_base = {
            strip_pad_suffix(r["case_id"]): r
            for r in runs_by_cell.get((m1, "RQ1a", "base"), [])
            if r.get("latency_s") is not None
        }
        m1_16 = {
            strip_pad_suffix(r["case_id"]): r
            for r in runs_by_cell.get((m1, "RQ1a", "pad_16384"), [])
            if r.get("latency_s") is not None
        }
        m2_base = {
            strip_pad_suffix(r["case_id"]): r
            for r in runs_by_cell.get((m2, "RQ1a", "base"), [])
            if r.get("latency_s") is not None
        }
        m2_16 = {
            strip_pad_suffix(r["case_id"]): r
            for r in runs_by_cell.get((m2, "RQ1a", "pad_16384"), [])
            if r.get("latency_s") is not None
        }
        cids = sorted(
            set(m1_base.keys())
            & set(m1_16.keys())
            & set(m2_base.keys())
            & set(m2_16.keys())
        )
        l_m1_16 = [m1_16[c]["latency_s"] for c in cids]
        l_m1_0 = [m1_base[c]["latency_s"] for c in cids]
        l_m2_16 = [m2_16[c]["latency_s"] for c in cids]
        l_m2_0 = [m2_base[c]["latency_s"] for c in cids]

        pt, ci, raw_p, _ = compute_ratio_of_ratios_with_pvalue(
            l_m1_16, l_m1_0, l_m2_16, l_m2_0, seed=seed, null_value=1.0
        )
        b_resamples = 10000
        p_floor = bool(raw_p <= (1.0 / (b_resamples + 1)) * 1.00001)
        cells_h1b = [(m1, "base"), (m1, "pad_16384"), (m2, "base"), (m2, "pad_16384")]
        return {
            "hypothesis": "H1(b)",
            "rq": "RQ1a",
            "family": "ratio_test",
            "contrast": f"G_{label_cand}(16K) / G_{label_ref}(16K)",
            "model": label_cand,
            "reference_model": label_ref,
            "level": "pad_16384 vs base",
            "null_val": 1.0,
            "direction": "<",
            "estimate": pt,
            "ci_low": ci[0],
            "ci_high": ci[1],
            "raw_p": raw_p,
            "p_floor": p_floor,
            "gated": False,
            "flag_low_valid": _get_flag_low_valid(cells_h1b),
            "notes": f"n_cases={len(cids)}",
        }

    for hosted_llm in ["DeepSeek-V4.1-Flash", "GLM-5.3-Flash", "Qwen3.8-Flash"]:
        h1b_raw_contrasts.append(
            _eval_h1b("Jev-1.13.0", hosted_llm, "Jev-1.13.0", hosted_llm)
        )
    # Self-hosted analogue: SemIf vs Qwen-JSON
    h1b_raw_contrasts.append(
        _eval_h1b(
            "SemIf-Qwen3.5-4B",
            "Qwen3.5-4B-JSON",
            "SemIf-Qwen3.5-4B",
            "Qwen3.5-4B-JSON",
        )
    )

    # Holm adjustment for H1(b) within RQ1a ratio test family
    h1b_pvals = [c["raw_p"] for c in h1b_raw_contrasts]
    h1b_adj = holm_adjust(h1b_pvals)
    for c, adj_p in zip(h1b_raw_contrasts, h1b_adj):
        c["holm_p"] = adj_p
        c["verdict"] = (
            "resolved"
            if _is_resolved(
                c["ci_low"], c["ci_high"], adj_p, c["null_val"], c["direction"]
            )
            else "not resolved"
        )

    # -----------------------------------------------------------------------
    # H2 (RQ1b): Per-request latency ratio R(k=8)/R(k=1)
    # -----------------------------------------------------------------------
    h2_raw_contrasts: list[dict[str, Any]] = []

    def _eval_h2(m1: str, m2: str, label_cand: str, label_ref: str) -> dict[str, Any]:
        m1_k1 = {
            extract_case_index(r["case_id"]): r["latency_s"] / 1.0
            for r in runs_by_cell.get((m1, "RQ1b", "k1"), [])
            if r.get("latency_s") is not None
        }
        m1_k8 = {
            extract_case_index(r["case_id"]): r["latency_s"] / 8.0
            for r in runs_by_cell.get((m1, "RQ1b", "k8"), [])
            if r.get("latency_s") is not None
        }
        m2_k1 = {
            extract_case_index(r["case_id"]): r["latency_s"] / 1.0
            for r in runs_by_cell.get((m2, "RQ1b", "k1"), [])
            if r.get("latency_s") is not None
        }
        m2_k8 = {
            extract_case_index(r["case_id"]): r["latency_s"] / 8.0
            for r in runs_by_cell.get((m2, "RQ1b", "k8"), [])
            if r.get("latency_s") is not None
        }
        idxs = sorted(
            set(m1_k1.keys()) & set(m1_k8.keys()) & set(m2_k1.keys()) & set(m2_k8.keys())
        )
        l_m1_k8 = [m1_k8[i] for i in idxs]
        l_m1_k1 = [m1_k1[i] for i in idxs]
        l_m2_k8 = [m2_k8[i] for i in idxs]
        l_m2_k1 = [m2_k1[i] for i in idxs]

        pt, ci, raw_p, _ = compute_ratio_of_ratios_with_pvalue(
            l_m1_k8, l_m1_k1, l_m2_k8, l_m2_k1, seed=seed, null_value=1.0
        )
        b_resamples = 10000
        p_floor = bool(raw_p <= (1.0 / (b_resamples + 1)) * 1.00001)
        cells_h2 = [(m1, "k1"), (m1, "k8"), (m2, "k1"), (m2, "k8")]
        return {
            "hypothesis": "H2",
            "rq": "RQ1b",
            "family": "ratio_test",
            "contrast": f"R_{label_cand}(8) / R_{label_ref}(8)",
            "model": label_cand,
            "reference_model": label_ref,
            "level": "k8 vs k1",
            "null_val": 1.0,
            "direction": "<",
            "estimate": pt,
            "ci_low": ci[0],
            "ci_high": ci[1],
            "raw_p": raw_p,
            "p_floor": p_floor,
            "gated": False,
            "flag_low_valid": _get_flag_low_valid(cells_h2),
            "notes": f"n_cases={len(idxs)}",
        }

    for hosted_llm in ["DeepSeek-V4.1-Flash", "GLM-5.3-Flash", "Qwen3.8-Flash"]:
        h2_raw_contrasts.append(
            _eval_h2("Jev-1.13.0", hosted_llm, "Jev-1.13.0", hosted_llm)
        )
    # Self-hosted analogue
    h2_raw_contrasts.append(
        _eval_h2(
            "SemIf-Qwen3.5-4B",
            "Qwen3.5-4B-JSON",
            "SemIf-Qwen3.5-4B",
            "Qwen3.5-4B-JSON",
        )
    )

    # Holm adjustment for H2 within RQ1b ratio test family
    h2_pvals = [c["raw_p"] for c in h2_raw_contrasts]
    h2_adj = holm_adjust(h2_pvals)
    for c, adj_p in zip(h2_raw_contrasts, h2_adj):
        c["holm_p"] = adj_p
        c["verdict"] = (
            "resolved"
            if _is_resolved(
                c["ci_low"], c["ci_high"], adj_p, c["null_val"], c["direction"]
            )
            else "not resolved"
        )

    # -----------------------------------------------------------------------
    # H3 (RQ2): Robustness & Unsafe-locality
    # -----------------------------------------------------------------------
    # 1. Omnibus Cochran's Q on unsafe across the six models per condition (8 conditions)
    h3_omnibus_contrasts: list[dict[str, Any]] = []
    # 2. Paired unsafe contrasts vs Jev-1.13.0 (8 conditions x 5 models = 40 contrasts)
    h3_paired_contrasts: list[dict[str, Any]] = []

    rq2_conds = EXPECTED_CONDITIONS["RQ2"]
    the_five = [m for m in THE_SIX_MODELS if m != "Jev-1.13.0"]

    for cond in rq2_conds:
        by_m_unsafe: dict[str, dict[str, bool]] = {}
        for m in THE_SIX_MODELS:
            rows = runs_by_cell.get((m, "RQ2", cond), [])
            by_m_unsafe[m] = {
                str(r["case_id"]): bool(r.get("unsafe_locality", False))
                for r in rows
            }
        common_cids = sorted(set.intersection(*(set(v) for v in by_m_unsafe.values())))

        # Omnibus Cochran's Q
        if common_cids:
            mat = [[int(by_m_unsafe[m][c]) for m in THE_SIX_MODELS] for c in common_cids]
            q_res = cochrans_q(mat)
        else:
            q_res = {"q": float("nan"), "p": float("nan"), "df": len(THE_SIX_MODELS) - 1, "n": 0}
        cells_h3_omni = [(m, cond) for m in THE_SIX_MODELS]
        h3_omnibus_contrasts.append(
            {
                "hypothesis": "H3_omnibus",
                "rq": "RQ2",
                "family": "omnibus_unsafe",
                "contrast": f"Cochran's Q on unsafe across six @ {cond}",
                "model": "The Six",
                "reference_model": "None",
                "level": cond,
                "null_val": 0.0,
                "direction": ">",
                "estimate": q_res["q"],
                "ci_low": q_res["q"],
                "ci_high": q_res["q"],
                "raw_p": q_res["p"],
                "p_floor": False,
                "gated": False,
                "flag_low_valid": _get_flag_low_valid(cells_h3_omni),
                "notes": f"df={q_res['df']}, n={q_res['n']}",
            }
        )

        # Paired contrasts vs Jev
        if common_cids:
            jev_un = [by_m_unsafe["Jev-1.13.0"][c] for c in common_cids]
            for m in the_five:
                m_un = [by_m_unsafe[m][c] for c in common_cids]
                diff = np.asarray(m_un, dtype=float) - np.asarray(jev_un, dtype=float)
                pt = float(np.mean(diff))
                ci = bootstrap_ci(diff, seed=seed, n_resamples=10000)
                p_val = exact_mcnemar_test(m_un, jev_un)
                cells_h3_pair = [(m, cond), ("Jev-1.13.0", cond)]
                h3_paired_contrasts.append(
                    {
                        "hypothesis": "H3_paired",
                        "rq": "RQ2",
                        "family": "unsafe_contrast",
                        "contrast": f"{m} - Jev-1.13.0 unsafe @ {cond}",
                        "model": m,
                        "reference_model": "Jev-1.13.0",
                        "level": cond,
                        "null_val": 0.0,
                        "direction": "!=",  # No directional prediction
                        "estimate": pt,
                        "ci_low": ci[0],
                        "ci_high": ci[1],
                        "raw_p": p_val,
                        "p_floor": False,
                        "gated": False,
                        "flag_low_valid": _get_flag_low_valid(cells_h3_pair),
                        "notes": f"n_cases={len(common_cids)}",
                    }
                )

    # Holm adjustment for H3 omnibus across 8 conditions
    h3_q_pvals = [c["raw_p"] for c in h3_omnibus_contrasts]
    h3_q_adj = holm_adjust(h3_q_pvals)
    for c, adj_p in zip(h3_omnibus_contrasts, h3_q_adj):
        c["holm_p"] = adj_p
        c["verdict"] = "resolved" if adj_p < 0.05 else "not resolved"

    # Holm adjustment for H3 paired contrasts across 8 x 5 = 40 contrasts
    h3_pair_pvals = [c["raw_p"] for c in h3_paired_contrasts]
    h3_pair_adj = holm_adjust(h3_pair_pvals)
    for c, adj_p in zip(h3_paired_contrasts, h3_pair_adj):
        c["holm_p"] = adj_p
        c["verdict"] = (
            "resolved"
            if _is_resolved(
                c["ci_low"], c["ci_high"], adj_p, c["null_val"], c["direction"]
            )
            else "not resolved"
        )

    # -----------------------------------------------------------------------
    # H4 (RQ3): Field scalability (F8 vs F4)
    # -----------------------------------------------------------------------
    # 1. Output tokens: median output tokens at F=8 minus F=4 > 0
    # 2. Latency growth ratio-of-ratios [p50_Jev(F8)/p50_Jev(F4)] / [p50_l(F8)/p50_l(F4)] < 1
    # Pooled over D, case-paired within each D, stratified bootstrap by D.
    h4_tokens_contrasts: list[dict[str, Any]] = []
    h4_latency_contrasts: list[dict[str, Any]] = []

    # Pairing cases by 4-digit case index within each D
    d_levels = ["low", "medium", "high"]

    # Hosted LLMs output tokens difference
    for m in ["DeepSeek-V4.1-Flash", "GLM-5.3-Flash", "Qwen3.8-Flash"]:
        diffs_tok: list[int] = []
        strata_tok: list[str] = []
        for d in d_levels:
            f4_rows = runs_by_cell.get((m, "RQ3", f"F4_{d}"), [])
            f8_rows = runs_by_cell.get((m, "RQ3", f"F8_{d}"), [])
            f4_by_idx = {
                extract_case_index(r["case_id"]): r["output_tokens"]
                for r in f4_rows
                if r.get("output_tokens") is not None
            }
            f8_by_idx = {
                extract_case_index(r["case_id"]): r["output_tokens"]
                for r in f8_rows
                if r.get("output_tokens") is not None
            }
            common = sorted(set(f4_by_idx.keys()) & set(f8_by_idx.keys()))
            for i in common:
                diffs_tok.append(f8_by_idx[i] - f4_by_idx[i])
                strata_tok.append(d)

        pt, ci, raw_p, _ = compute_median_difference_with_pvalue(
            diffs_tok,
            seed=seed,
            n_resamples=10000,
            strata=strata_tok,
            null_value=0.0,
        )
        p_floor = bool(raw_p <= (1.0 / 10001) * 1.00001)
        cells_h4_tok = [(m, f"F4_{d}") for d in d_levels] + [(m, f"F8_{d}") for d in d_levels]
        h4_tokens_contrasts.append(
            {
                "hypothesis": "H4_tokens",
                "rq": "RQ3",
                "family": "tokens_contrast",
                "contrast": f"{m} output tokens F8 - F4",
                "model": m,
                "reference_model": "None",
                "level": "F8 vs F4 pooled over D",
                "null_val": 0.0,
                "direction": ">",
                "estimate": pt,
                "ci_low": ci[0],
                "ci_high": ci[1],
                "raw_p": raw_p,
                "p_floor": p_floor,
                "gated": False,
                "flag_low_valid": _get_flag_low_valid(cells_h4_tok),
                "notes": f"n_pairs={len(diffs_tok)}",
            }
        )

    # Holm adjustment for H4 tokens family
    h4_tok_pvals = [c["raw_p"] for c in h4_tokens_contrasts]
    h4_tok_adj = holm_adjust(h4_tok_pvals)
    for c, adj_p in zip(h4_tokens_contrasts, h4_tok_adj):
        c["holm_p"] = adj_p
        c["verdict"] = (
            "resolved"
            if _is_resolved(
                c["ci_low"], c["ci_high"], adj_p, c["null_val"], c["direction"]
            )
            else "not resolved"
        )

    # H4 Latency ratio-of-ratios
    def _eval_h4_lat(m1: str, m2: str, label_cand: str, label_ref: str) -> dict[str, Any]:
        l_m1_8, l_m1_4 = [], []
        l_m2_8, l_m2_4 = [], []
        strata: list[str] = []
        for d in d_levels:
            m1_f4 = {
                extract_case_index(r["case_id"]): r["latency_s"]
                for r in runs_by_cell.get((m1, "RQ3", f"F4_{d}"), [])
                if r.get("latency_s") is not None
            }
            m1_f8 = {
                extract_case_index(r["case_id"]): r["latency_s"]
                for r in runs_by_cell.get((m1, "RQ3", f"F8_{d}"), [])
                if r.get("latency_s") is not None
            }
            m2_f4 = {
                extract_case_index(r["case_id"]): r["latency_s"]
                for r in runs_by_cell.get((m2, "RQ3", f"F4_{d}"), [])
                if r.get("latency_s") is not None
            }
            m2_f8 = {
                extract_case_index(r["case_id"]): r["latency_s"]
                for r in runs_by_cell.get((m2, "RQ3", f"F8_{d}"), [])
                if r.get("latency_s") is not None
            }
            common = sorted(
                set(m1_f4.keys())
                & set(m1_f8.keys())
                & set(m2_f4.keys())
                & set(m2_f8.keys())
            )
            for i in common:
                l_m1_8.append(m1_f8[i])
                l_m1_4.append(m1_f4[i])
                l_m2_8.append(m2_f8[i])
                l_m2_4.append(m2_f4[i])
                strata.append(d)

        pt, ci, raw_p, _ = compute_ratio_of_ratios_with_pvalue(
            l_m1_8, l_m1_4, l_m2_8, l_m2_4, seed=seed, strata=strata, null_value=1.0
        )
        p_floor = bool(raw_p <= (1.0 / 10001) * 1.00001)
        cells_h4_lat = (
            [(m1, f"F4_{d}") for d in d_levels]
            + [(m1, f"F8_{d}") for d in d_levels]
            + [(m2, f"F4_{d}") for d in d_levels]
            + [(m2, f"F8_{d}") for d in d_levels]
        )
        return {
            "hypothesis": "H4_latency",
            "rq": "RQ3",
            "family": "ratio_test",
            "contrast": f"G_{label_cand}(F8/F4) / G_{label_ref}(F8/F4)",
            "model": label_cand,
            "reference_model": label_ref,
            "level": "F8 vs F4 pooled over D",
            "null_val": 1.0,
            "direction": "<",
            "estimate": pt,
            "ci_low": ci[0],
            "ci_high": ci[1],
            "raw_p": raw_p,
            "p_floor": p_floor,
            "gated": False,
            "flag_low_valid": _get_flag_low_valid(cells_h4_lat),
            "notes": f"n_pairs={len(l_m1_8)} (case-paired within D: 300 low, 300 med, 300 high)",
        }

    for hosted_llm in ["DeepSeek-V4.1-Flash", "GLM-5.3-Flash", "Qwen3.8-Flash"]:
        h4_latency_contrasts.append(
            _eval_h4_lat("Jev-1.13.0", hosted_llm, "Jev-1.13.0", hosted_llm)
        )
    # Self-hosted analogue: SemIf vs Qwen-JSON
    h4_latency_contrasts.append(
        _eval_h4_lat(
            "SemIf-Qwen3.5-4B",
            "Qwen3.5-4B-JSON",
            "SemIf-Qwen3.5-4B",
            "Qwen3.5-4B-JSON",
        )
    )

    # Holm adjustment for H4 latency family
    h4_lat_pvals = [c["raw_p"] for c in h4_latency_contrasts]
    h4_lat_adj = holm_adjust(h4_lat_pvals)
    for c, adj_p in zip(h4_latency_contrasts, h4_lat_adj):
        c["holm_p"] = adj_p
        c["verdict"] = (
            "resolved"
            if _is_resolved(
                c["ci_low"], c["ci_high"], adj_p, c["null_val"], c["direction"]
            )
            else "not resolved"
        )

    # -----------------------------------------------------------------------
    # H5 (RQ4b): Zero-training seen-unseen accuracy gap < 10 pp at churn25/churn50
    # -----------------------------------------------------------------------
    h5_raw_contrasts: list[dict[str, Any]] = []

    h5_eval_models = THE_SIX_MODELS + [REFERENCE_MODEL, "MiniLM-Reranker"]
    for cond in ["churn25", "churn50"]:
        cases_map = cases_by_rq_cond.get("RQ4", {}).get(cond, {})
        for m in h5_eval_models:
            rows = runs_by_cell.get((m, "RQ4", cond), [])
            if not rows:
                continue
            gap, ci, raw_p, upper = compute_catalog_gap_with_pvalue(
                rows,
                cases_map=cases_map,
                null_value=0.10,
                seed=seed,
                n_resamples=10000,
            )
            p_floor = bool(raw_p <= (1.0 / 10001) * 1.00001)
            cells_h5 = [(m, cond)]
            h5_raw_contrasts.append(
                {
                    "hypothesis": "H5",
                    "rq": "RQ4",
                    "family": "catalog_gap",
                    "contrast": f"{m} seen - unseen gap @ {cond}",
                    "model": m,
                    "reference_model": "None",
                    "level": cond,
                    "null_val": 0.10,
                    "direction": "upper_bound_lt",
                    "estimate": gap,
                    "ci_low": ci[0] if ci else float("nan"),
                    "ci_high": ci[1] if ci else float("nan"),
                    "raw_p": raw_p,
                    "p_floor": p_floor,
                    "gated": False,
                    "flag_low_valid": _get_flag_low_valid(cells_h5),
                    "notes": f"upper_ci95={upper:.4f}" if upper is not None else "",
                }
            )

    # Holm adjustment for H5 within RQ4 catalog gap family
    h5_pvals = [c["raw_p"] for c in h5_raw_contrasts]
    h5_adj = holm_adjust(h5_pvals)
    for c, adj_p in zip(h5_raw_contrasts, h5_adj):
        c["holm_p"] = adj_p
        c["verdict"] = (
            "resolved"
            if _is_resolved(
                c["ci_low"], c["ci_high"], adj_p, c["null_val"], c["direction"]
            )
            else "not resolved"
        )

    # Assemble all hypotheses
    all_contrasts = (
        h1a_raw_contrasts
        + h1b_raw_contrasts
        + h2_raw_contrasts
        + h3_omnibus_contrasts
        + h3_paired_contrasts
        + h4_tokens_contrasts
        + h4_latency_contrasts
        + h5_raw_contrasts
    )
    hypotheses_rows.extend(all_contrasts)

    # -----------------------------------------------------------------------
    # Format hypotheses.md
    # -----------------------------------------------------------------------
    md_lines: list[str] = [
        "# Confirmatory Hypotheses Evaluation (H1–H5)",
        "",
        "Evaluation against pre-registered analysis plan (`analysis-plan.md`, frozen 2026-09-24).",
        "Decision rule: resolved = 95% CI excludes null value AND Holm-adjusted p < 0.05.",
        "",
        "## Summary of Confirmatory Status",
        "",
        "| Hypothesis | Target RQ | Family | Contrasts Tested | Resolved | Notes |",
        "|---|---|---|---|---|---|",
        f"| **H1(a)** | RQ1a | Latency difference | {len(h1a_valid)} | {sum(1 for c in h1a_valid if c['verdict'] == 'resolved')} | Gated out: {len(gated_out_h1a)} levels |",
        f"| **H1(b)** | RQ1a | Ratio-of-ratios | {len(h1b_raw_contrasts)} | {sum(1 for c in h1b_raw_contrasts if c['verdict'] == 'resolved')} | {_unresolved_note(h1b_raw_contrasts)} |",
        f"| **H2** | RQ1b | Per-request ratio-of-ratios | {len(h2_raw_contrasts)} | {sum(1 for c in h2_raw_contrasts if c['verdict'] == 'resolved')} | {_unresolved_note(h2_raw_contrasts)} |",
        f"| **H3 (Omnibus)** | RQ2 | Cochran's Q on unsafe | {len(h3_omnibus_contrasts)} | {sum(1 for c in h3_omnibus_contrasts if c['verdict'] == 'resolved')} | {_unresolved_note(h3_omnibus_contrasts)} |",
        f"| **H3 (Paired)** | RQ2 | Unsafe difference vs Jev | {len(h3_paired_contrasts)} | {sum(1 for c in h3_paired_contrasts if c['verdict'] == 'resolved')} | {_unresolved_note(h3_paired_contrasts)} |",
        f"| **H4 (Tokens)** | RQ3 | Output tokens diff | {len(h4_tokens_contrasts)} | {sum(1 for c in h4_tokens_contrasts if c['verdict'] == 'resolved')} | {_unresolved_note(h4_tokens_contrasts)} |",
        f"| **H4 (Latency)** | RQ3 | Latency ratio-of-ratios | {len(h4_latency_contrasts)} | {sum(1 for c in h4_latency_contrasts if c['verdict'] == 'resolved')} | {_unresolved_note(h4_latency_contrasts)} |",
        f"| **H5** | RQ4b | Catalog gap churn25/50 | {len(h5_raw_contrasts)} | {sum(1 for c in h5_raw_contrasts if c['verdict'] == 'resolved')} | {_unresolved_note(h5_raw_contrasts)} |",
        "",
        "## Gating Status (H1a Valid Rate >= 90%)",
        "",
    ]

    if gated_out_h1a:
        for g in gated_out_h1a:
            md_lines.append(f"- Gated out: {g}")
    else:
        md_lines.append("- No levels were gated out: all evaluated models achieved >= 90% valid rate.")
    md_lines.append("")

    # Flagged cells (< 50% valid rate)
    md_lines.append(generate_flagged_cells_md(runs_by_cell, models=ALL_EVAL_MODELS))

    # Tables per hypothesis
    def _render_table(title: str, contrasts: list[dict[str, Any]]) -> None:
        md_lines.append(f"### {title}")
        md_lines.append("")
        md_lines.append(
            "| Contrast | Level | Estimate | 95% CI | Raw p | Holm p | Verdict | Notes |"
        )
        md_lines.append("|---|---|---|---|---|---|---|---|")
        for c in contrasts:
            est = c.get("estimate")
            est_str = (
                f"{est:.4f}"
                if est is not None and not np.isnan(est)
                else "NA"
            )
            ci_low = c.get("ci_low")
            ci_high = c.get("ci_high")
            ci_str = (
                f"[{ci_low:.4f}, {ci_high:.4f}]"
                if ci_low is not None
                and ci_high is not None
                and not (np.isnan(ci_low) or np.isnan(ci_high))
                else "NA"
            )
            raw_p = c.get("raw_p")
            holm_p = c.get("holm_p")
            if c.get("p_floor"):
                raw_p_str = "< 1e-4"
                if holm_p is not None and not np.isnan(holm_p) and holm_p < 1e-4:
                    holm_p_str = "< 1e-4"
                elif holm_p is not None and not np.isnan(holm_p):
                    holm_p_str = f"{holm_p:.4e}"
                else:
                    holm_p_str = "NA"
            else:
                raw_p_str = (
                    f"{raw_p:.4e}"
                    if raw_p is not None and not np.isnan(raw_p)
                    else "NA"
                )
                holm_p_str = (
                    f"{holm_p:.4e}"
                    if holm_p is not None and not np.isnan(holm_p)
                    else "NA"
                )

            v_str = c["verdict"]
            if c.get("flag_low_valid"):
                v_str = f"{v_str} (flagged: valid <50%)"

            md_lines.append(
                f"| {c['contrast']} | {c['level']} | {est_str} | {ci_str} | {raw_p_str} | {holm_p_str} | **{v_str}** | {c['notes']} |"
            )
        md_lines.append("")

    _render_table("H1(a): Paired Latency Difference (LLM - Jev > 0)", h1a_raw_contrasts)
    _render_table("H1(b): Latency Growth Ratio-of-Ratios (< 1.0)", h1b_raw_contrasts)
    _render_table("H2: Per-Request Latency Ratio-of-Ratios (k=8 vs k=1, < 1.0)", h2_raw_contrasts)
    _render_table("H3: Omnibus Cochran's Q on Unsafe Locality across The Six", h3_omnibus_contrasts)
    _render_table("H3: Paired Unsafe Differences vs Jev-1.13.0", h3_paired_contrasts)
    _render_table("H4: Output Tokens Growth (F=8 minus F=4 > 0)", h4_tokens_contrasts)
    _render_table("H4: Latency Growth Ratio-of-Ratios Pooled over D (< 1.0)", h4_latency_contrasts)
    _render_table("H5: Seen - Unseen Top-1 Accuracy Gap (< 10 pp)", h5_raw_contrasts)

    clf_rows, _ = generate_h5_classifier_reference(
        runs_by_cell, cases_by_rq_cond, runs_dir=runs_dir, seed=seed
    )
    md_lines.append(generate_h5_classifier_reference_md(clf_rows))

    hypotheses_md = "\n".join(md_lines)
    return hypotheses_rows, hypotheses_md


def generate_hypotheses_csv(hypotheses_rows: list[dict[str, Any]]) -> str:
    """Serialize hypotheses rows to CSV."""
    fieldnames = [
        "hypothesis",
        "rq",
        "family",
        "contrast",
        "model",
        "reference_model",
        "level",
        "null_val",
        "direction",
        "estimate",
        "ci_low",
        "ci_high",
        "raw_p",
        "holm_p",
        "p_floor",
        "gated",
        "verdict",
        "flag_low_valid",
        "notes",
    ]
    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(hypotheses_rows)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Sensitivity & Report Generation
# ---------------------------------------------------------------------------

def generate_sensitivity_table(
    runs_by_cell: dict[tuple[str, str, str], list[dict[str, Any]]],
    sensitivity_dir: str | Path,
    seed: int = 20260924,
) -> tuple[list[dict[str, Any]], str]:
    """Generate sensitivity.csv: D-7 sensitivity re-run next to its primary cell."""
    sensitivity_dir = Path(sensitivity_dir)
    rerun_ledger = (
        sensitivity_dir / "RQ3" / "F4_low__GLM-5.3-Flash__rerun1" / "ledger.jsonl"
    )
    out_rows: list[dict[str, Any]] = []

    if rerun_ledger.exists():
        rerun_rows = load_ledger_rows(rerun_ledger)
        primary_rows = runs_by_cell.get(("GLM-5.3-Flash", "RQ3", "F4_low"), [])

        agg_p = aggregate_metrics(primary_rows, seed=seed)
        agg_r = aggregate_metrics(rerun_rows, seed=seed)

        metrics = [
            ("n", "n", agg_p["n"], agg_r["n"]),
            ("em", "EM", agg_p["em"], agg_r["em"]),
            (
                "em_ci95",
                "EM 95% CI",
                f"[{agg_p['em_ci95'][0]:.4f}, {agg_p['em_ci95'][1]:.4f}]",
                f"[{agg_r['em_ci95'][0]:.4f}, {agg_r['em_ci95'][1]:.4f}]",
            ),
            (
                "macro_field_accuracy",
                "Macro Field Acc",
                agg_p["macro_field_accuracy"],
                agg_r["macro_field_accuracy"],
            ),
            ("valid_rate", "Valid Rate", agg_p["valid_rate"], agg_r["valid_rate"]),
            ("unsafe_rate", "Unsafe Rate", agg_p["unsafe_rate"], agg_r["unsafe_rate"]),
            (
                "spurious_rate",
                "Spurious Rate",
                agg_p["spurious_rate"],
                agg_r["spurious_rate"],
            ),
            ("missed_rate", "Missed Rate", agg_p["missed_rate"], agg_r["missed_rate"]),
            ("latency_p50", "Latency p50 (s)", agg_p["latency_p50"], agg_r["latency_p50"]),
            ("latency_p95", "Latency p95 (s)", agg_p["latency_p95"], agg_r["latency_p95"]),
            ("latency_p99", "Latency p99 (s)", agg_p["latency_p99"], agg_r["latency_p99"]),
        ]

        for m_id, m_name, p_val, r_val in metrics:
            delta_val = ""
            if isinstance(p_val, (int, float)) and isinstance(r_val, (int, float)):
                delta_val = f"{r_val - p_val:+.4f}"
            out_rows.append(
                {
                    "model": "GLM-5.3-Flash",
                    "rq": "RQ3",
                    "condition": "F4_low",
                    "metric_id": m_id,
                    "metric_name": m_name,
                    "primary_run": p_val,
                    "sensitivity_rerun1": r_val,
                    "delta_rerun_minus_primary": delta_val,
                }
            )

    fieldnames = [
        "model",
        "rq",
        "condition",
        "metric_id",
        "metric_name",
        "primary_run",
        "sensitivity_rerun1",
        "delta_rerun_minus_primary",
    ]
    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(out_rows)
    return out_rows, buf.getvalue()


def check_oracle_control(
    runs_dir: str | Path,
) -> tuple[bool, list[tuple[str, str, float, int]], str]:
    """Check positive control Oracle interpreter across all reference ledgers.

    Returns (all_em_one, details, markdown_table).
    """
    runs_dir = Path(runs_dir)
    ref_dir = runs_dir / "_reference"
    details: list[tuple[str, str, float, int]] = []
    all_one = True

    lines = ["| RQ | Condition | Oracle Rows | Oracle EM | Status |", "|---|---|---|---|---|"]

    if ref_dir.exists():
        for rq, conds in EXPECTED_CONDITIONS.items():
            for cond in conds:
                p = ref_dir / rq / cond / "ledger.jsonl"
                if p.exists():
                    rows = [
                        r
                        for r in load_ledger_rows(p)
                        if r.get("repeat", 0) == 0 and r.get("model") == "Oracle"
                    ]
                    n = len(rows)
                    em_rate = float(np.mean([bool(r.get("em", False)) for r in rows])) if n else 0.0
                    ok = (em_rate == 1.0) and (n == 300)
                    if not ok:
                        all_one = False
                    status_str = "PASS (1.000)" if ok else f"**FLAG (EM={em_rate:.3f}, n={n})**"
                    details.append((rq, cond, em_rate, n))
                    lines.append(f"| {rq} | {cond} | {n} | {em_rate:.4f} | {status_str} |")

    return all_one, details, "\n".join(lines)



def _unresolved_note(contrasts: list[dict[str, Any]]) -> str:
    """Summary-table note computed from the verdicts: which tested contrasts are not resolved."""
    tested = [c for c in contrasts if c.get("verdict") not in ("gated out", "reranker not run")]
    bad = [str(c.get("contrast")) for c in tested if not str(c.get("verdict", "")).startswith("resolved")]
    return "all resolved" if not bad else "not resolved: " + "; ".join(bad)

def generate_report_md(
    coverage_matrix_md: str,
    oracle_table_md: str,
    cells_rows: list[dict[str, Any]],
    comms_rows: list[dict[str, Any]],
    all_oracle_passed: bool,
    runs_by_cell: dict[tuple[str, str, str], list[dict[str, Any]]] | None = None,
    flagged_cells_md: str | None = None,
) -> str:
    """Generate comprehensive report.md."""
    lines: list[str] = [
        "# EXP-2026-001 Analysis Report & Addendum A1 Synthesis",
        "",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} | **Protocol:** Frozen analysis-plan.md + Addendum A1",
        "",
        "## 1. Executive Summary",
        "",
        "- **Coverage**: see Section 2 (strict mode exits non-zero if any expected cell is missing or has n != 300).",
        f"- **Oracle Positive Control**: {'PASSED (EM = 1.0 in every cell across all reference runs)' if all_oracle_passed else 'FLAGGED (some cells deviated from EM 1.0)'}.",
        "- **Confirmatory results**: see `hypotheses.md` (every verdict there is computed from `hypotheses.csv`).",
        "",
        "## 2. Coverage Matrix",
        "",
        coverage_matrix_md,
        "",
        "## 3. Oracle Positive Control Results",
        "",
        "Per protocol, the Oracle interpreter must achieve EM 1.0 in every evaluated cell as a positive control.",
        "",
        oracle_table_md,
        "",
        "## 4. Short Summary Tables by Research Question",
        "",
    ]

    # Helper to format a sub-table from cells_rows
    def _make_rq_table(rq_name: str, metric_keys: list[tuple[str, str]]) -> str:
        rows = [r for r in cells_rows if r["rq"] == rq_name]
        headers = ["Condition", "Model"] + [m[1] for m in metric_keys]
        t_lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for r in rows:
            vals = [str(r["condition"]), str(r["model"])]
            for k, _ in metric_keys:
                val = r.get(k, "")
                if isinstance(val, float):
                    vals.append(f"{val:.4f}")
                else:
                    vals.append(str(val))
            t_lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(t_lines)

    lines.extend(
        [
            "### RQ1a: Input Length Scalability",
            _make_rq_table(
                "RQ1a",
                [
                    ("em", "EM"),
                    ("valid_rate", "Valid"),
                    ("latency_p50", "p50 (s)"),
                    ("latency_p95", "p95 (s)"),
                ],
            ),
            "",
            "### RQ1b: Batching & Bundle Scalability",
            _make_rq_table(
                "RQ1b",
                [
                    ("rq1b_request_em", "Request EM"),
                    ("rq1b_message_all_correct", "Msg EM"),
                    ("latency_p50", "p50 (s)"),
                ],
            ),
            "",
            "### RQ2: Environmental & Syntactic Robustness",
            _make_rq_table(
                "RQ2",
                [
                    ("em", "EM"),
                    ("unsafe_rate", "Unsafe Rate"),
                    ("valid_rate", "Valid"),
                ],
            ),
            "",
            "### RQ3: Schema & Field Scalability (F=4, 6, 8)",
            _make_rq_table(
                "RQ3",
                [
                    ("em", "EM"),
                    ("output_tokens_median", "Out Tok p50"),
                    ("latency_p50", "p50 (s)"),
                ],
            ),
            "",
            "### RQ4: Dynamic Catalog & Churn",
            _make_rq_table(
                "RQ4",
                [
                    ("seen_top1", "Seen Top-1"),
                    ("unseen_top1", "Unseen Top-1"),
                    ("unsupported_f1", "Unsupp F1"),
                ],
            ),
            "",
            "## 5. Communications-Community Metrics Summary (Addendum A1)",
            "",
            "Descriptive statistics aligned with communications/networking venues (availability, SLA violation, jitter, single-client throughput, goodput, energy):",
            "",
        ]
    )

    # Comms summary sample
    comms_sample_headers = [
        "Model",
        "RQ/Cond",
        "Avail",
        "SLA Viol",
        "Tput (dec/s)",
        "Goodput (corr/s)",
        "Jitter IQR (s)",
        "Energy (J/dec)",
    ]
    c_lines = [
        "| " + " | ".join(comms_sample_headers) + " |",
        "| " + " | ".join(["---"] * len(comms_sample_headers)) + " |",
    ]
    for r in comms_rows[:35]:  # display a representative sample
        e_str = (
            f"{r['energy_j_per_decision']:.2f}"
            if isinstance(r.get("energy_j_per_decision"), (int, float))
            else "-"
        )
        avail_str = f"{r['availability']:.3f}" if isinstance(r.get("availability"), (int, float)) else "-"
        sla_str = f"{r['sla_violation_rate']:.3f}" if isinstance(r.get("sla_violation_rate"), (int, float)) else "-"
        tput_str = f"{r['throughput_decisions_per_s']:.2f}" if isinstance(r.get("throughput_decisions_per_s"), (int, float)) else "-"
        gput_str = f"{r['goodput_correct_per_s']:.2f}" if isinstance(r.get("goodput_correct_per_s"), (int, float)) else "-"
        jitter_str = f"{r['jitter_iqr_s']:.4f}" if isinstance(r.get("jitter_iqr_s"), (int, float)) else "-"
        c_lines.append(
            f"| {r['model']} | {r['rq']}/{r['condition']} | {avail_str} | {sla_str} | {tput_str} | {gput_str} | {jitter_str} | {e_str} |"
        )
    lines.append("\n".join(c_lines))
    lines.append("")

    if flagged_cells_md is None and runs_by_cell is not None:
        flagged_cells_md = generate_flagged_cells_md(runs_by_cell)
    if flagged_cells_md:
        lines.append(flagged_cells_md)

    return "\n".join(lines)
