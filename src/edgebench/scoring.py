"""Scoring, statistical testing, and aggregation for Edgebench."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import warnings

import numpy as np
from scipy.stats import binomtest, chi2, wilcoxon

from src.edgebench.contract import Case
from src.edgebench.interpreters.base import Decision
from src.edgebench.manifest import load_manifest

# Manifest `model` prefix of the Laya checkpoint (every device); its server leaves input_tokens null and
# scripts/eb_laya_tokens.py writes them to the <ledger>.laya_tokens.jsonl sidecar.
LAYA_MODEL_PREFIX = "laya-typed-decisions@"


class ScoreResult(tuple):
    """Result of scoring a single decision against case truth.

    Subclasses tuple for backwards compatibility (unpacks as 7 elements),
    while also exposing attributes and per-request correctness lists.
    """
    correct: dict[str, bool]
    em: bool
    unsafe_locality: bool
    spurious_count: int
    missed_count: int
    unspecified_truth_count: int
    specified_truth_count: int
    request_em: list[bool]
    request_field_correct: list[dict[str, bool]]

    def __new__(
        cls,
        correct: dict[str, bool],
        em: bool,
        unsafe_locality: bool,
        spurious_count: int,
        missed_count: int,
        unspecified_truth_count: int,
        specified_truth_count: int,
        request_em: list[bool] | None = None,
        request_field_correct: list[dict[str, bool]] | None = None,
    ) -> ScoreResult:
        instance = super().__new__(
            cls,
            (
                correct,
                em,
                unsafe_locality,
                spurious_count,
                missed_count,
                unspecified_truth_count,
                specified_truth_count,
            ),
        )
        instance.correct = correct
        instance.em = em
        instance.unsafe_locality = unsafe_locality
        instance.spurious_count = spurious_count
        instance.missed_count = missed_count
        instance.unspecified_truth_count = unspecified_truth_count
        instance.specified_truth_count = specified_truth_count
        instance.request_em = [] if request_em is None else request_em
        instance.request_field_correct = (
            [] if request_field_correct is None else request_field_correct
        )
        return instance

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, int):
            if item == 7:
                return self.request_em
            elif item == 8:
                return self.request_field_correct
        return super().__getitem__(item)


def score_decision(decision: Decision, case: Case) -> ScoreResult:
    """Score a single decision against case truth.

    Returns:
        ScoreResult tuple: (correct_dict, em, unsafe_locality, spurious_count, missed_count, unspecified_truth_count, specified_truth_count)
        with .request_em and .request_field_correct attributes.
    """
    fields = case.fields
    k = case.bundle_size
    truth = case.truth

    # Default counters
    unspecified_truth_count = 0
    specified_truth_count = 0
    for req_truth in truth:
        for f in fields:
            if req_truth.get(f, "unspecified") == "unspecified":
                unspecified_truth_count += 1
            else:
                specified_truth_count += 1

    labels = decision.labels if decision.labels else []

    # Check unsafe locality from labels
    unsafe_locality = False
    for i in range(k):
        pred_req = labels[i] if i < len(labels) else {}
        truth_req = truth[i] if i < len(truth) else {}
        if (
            pred_req.get("locality") == "remote_allowed"
            and truth_req.get("locality") != "remote_allowed"
        ):
            unsafe_locality = True
            break

    # Spurious and missed counts: computed from labels even for invalid decisions
    spurious_count = 0
    missed_count = 0
    for i in range(k):
        pred_req = labels[i] if i < len(labels) else {}
        truth_req = truth[i] if i < len(truth) else {}
        for f in fields:
            t_val = truth_req.get(f, "unspecified")
            p_val = pred_req.get(f, "unspecified")
            if t_val == "unspecified" and p_val != "unspecified":
                spurious_count += 1
            elif t_val != "unspecified" and p_val == "unspecified":
                missed_count += 1

    request_field_correct: list[dict[str, bool]] = []
    request_em: list[bool] = []

    if not decision.valid or len(labels) != k:
        correct = {f: False for f in fields}
        em = False
        request_em = [False] * k
        request_field_correct = [{f: False for f in fields} for _ in range(k)]
    else:
        for i in range(k):
            pred_req = labels[i]
            truth_req = truth[i]
            req_fc = {f: (pred_req.get(f) == truth_req.get(f)) for f in fields}
            request_field_correct.append(req_fc)
            request_em.append(all(req_fc.values()))

        correct = {
            f: all(request_field_correct[i][f] for i in range(k))
            for f in fields
        }
        em = all(request_em)

    return ScoreResult(
        correct=correct,
        em=em,
        unsafe_locality=unsafe_locality,
        spurious_count=spurious_count,
        missed_count=missed_count,
        unspecified_truth_count=unspecified_truth_count,
        specified_truth_count=specified_truth_count,
        request_em=request_em,
        request_field_correct=request_field_correct,
    )


def bootstrap_ci(
    values: list[float] | np.ndarray,
    n_resamples: int = 10000,
    seed: int = 20260924,
    stat_fn: Any = np.mean,
) -> tuple[float, float]:
    """Compute 95% bootstrap confidence interval with fixed seed."""
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return (0.0, 0.0)
    try:
        point = float(stat_fn(arr))
    except TypeError:
        point = float(stat_fn(arr, axis=None))

    if len(arr) == 1 or np.all(arr == arr[0]):
        return (point, point)

    rng = np.random.default_rng(seed)
    n = len(arr)
    indices = rng.integers(0, n, size=(n_resamples, n))
    try:
        samples = stat_fn(arr[indices], axis=1)
    except TypeError:
        samples = np.array([stat_fn(arr[idx]) for idx in indices], dtype=float)

    ci_low = float(np.percentile(samples, 2.5))
    ci_high = float(np.percentile(samples, 97.5))

    # Guarantee point estimate is within CI
    ci_low = min(ci_low, point)
    ci_high = max(ci_high, point)
    return (ci_low, ci_high)


def _bootstrap_ratio_ci(
    numerators: list[float] | np.ndarray,
    denominators: list[float] | np.ndarray,
    n_resamples: int = 10000,
    seed: int = 20260924,
) -> tuple[float, float]:
    """Compute 95% bootstrap CI for ratio of sums: sum(numerators) / sum(denominators)."""
    nums = np.asarray(numerators, dtype=float)
    dens = np.asarray(denominators, dtype=float)
    total_num = float(np.sum(nums))
    total_den = float(np.sum(dens))
    point = total_num / total_den if total_den > 0 else 0.0
    if total_den == 0 or len(nums) == 0:
        return (0.0, 0.0)
    if len(nums) == 1 or (np.all(nums == nums[0]) and np.all(dens == dens[0])):
        return (point, point)

    rng = np.random.default_rng(seed)
    n = len(nums)
    indices = rng.integers(0, n, size=(n_resamples, n))
    num_samples = np.sum(nums[indices], axis=1)
    den_samples = np.sum(dens[indices], axis=1)
    ratios = np.where(den_samples > 0, num_samples / den_samples, 0.0)

    ci_low = float(np.percentile(ratios, 2.5))
    ci_high = float(np.percentile(ratios, 97.5))
    ci_low = min(ci_low, point)
    ci_high = max(ci_high, point)
    return (ci_low, ci_high)


def exact_mcnemar_test(
    model_em: list[bool] | np.ndarray, ref_em: list[bool] | np.ndarray
) -> float:
    """Exact McNemar test using scipy.stats.binomtest on discordant pairs."""
    m_arr = np.asarray(model_em, dtype=bool)
    r_arr = np.asarray(ref_em, dtype=bool)
    if len(m_arr) != len(r_arr):
        raise ValueError("Model and reference arrays must have the same length")

    # b = model correct, ref incorrect; c = model incorrect, ref correct
    b = int(np.sum(m_arr & ~r_arr))
    c = int(np.sum(~m_arr & r_arr))

    if b + c == 0:
        return 1.0

    res = binomtest(b, n=b + c, p=0.5)
    return float(res.pvalue)


def cochrans_q(matrix: list[list[int]] | np.ndarray) -> dict[str, Any]:
    """Cochran's Q test on a case-aligned binary matrix (rows = cases, columns = k interpreters).

    Q = (k-1) * (k * sum_j C_j^2 - N^2) / (k * N - sum_i R_i^2), with C_j column totals,
    R_i row totals and N the grand total; p from the chi-square distribution with k-1 df.
    When every case row is constant (denominator 0), Q = 0 and p = 1.
    """
    x = np.asarray(matrix, dtype=float)
    if x.ndim != 2 or x.shape[1] < 2:
        raise ValueError("Cochran's Q needs a 2-D matrix with at least two interpreter columns")
    if not np.all((x == 0) | (x == 1)):
        raise ValueError("Cochran's Q needs a binary matrix")
    n, k = x.shape
    col = x.sum(axis=0)
    row = x.sum(axis=1)
    total = float(x.sum())
    denom = k * total - float(np.sum(row ** 2))
    df = k - 1
    if denom == 0:
        return {"q": 0.0, "df": df, "p": 1.0, "n": n, "k": k}
    q = (k - 1) * (k * float(np.sum(col ** 2)) - total ** 2) / denom
    return {"q": float(q), "df": df, "p": float(chi2.sf(q, df)), "n": n, "k": k}


def cochrans_q_pvalues_by_condition(
    rows_by_condition: dict[str, dict[str, list[dict[str, Any]]]],
    metric: str = "em",
) -> dict[str, float]:
    """Per-condition Cochran's Q p-values across interpreters, ready for holm_adjust.

    rows_by_condition maps condition -> interpreter -> ledger rows. Only repeat-0 rows are used;
    the matrix is aligned on the case ids present for every interpreter. `metric` is the per-case
    binary indicator ("em" or "unsafe_locality"); invalid rows keep their scored value.
    """
    if metric not in ("em", "unsafe_locality"):
        raise ValueError(f"Unsupported metric for Cochran's Q: {metric}")
    pvalues: dict[str, float] = {}
    for condition, by_model in rows_by_condition.items():
        models = sorted(by_model)
        per_model = {
            m: {str(r["case_id"]): bool(r.get(metric, False)) for r in by_model[m] if r.get("repeat", 0) == 0}
            for m in models
        }
        common = set.intersection(*(set(v) for v in per_model.values())) if per_model else set()
        case_ids = sorted(common)
        matrix = [[int(per_model[m][cid]) for m in models] for cid in case_ids]
        if not matrix:
            raise ValueError(f"No case-aligned rows for condition '{condition}'")
        pvalues[condition] = cochrans_q(matrix)["p"]
    return pvalues


def paired_wilcoxon_test(
    latencies_model: list[float] | np.ndarray,
    latencies_ref: list[float] | np.ndarray,
) -> float:
    """Paired Wilcoxon signed-rank test on latency."""
    m_lats = np.asarray(latencies_model, dtype=float)
    r_lats = np.asarray(latencies_ref, dtype=float)
    if len(m_lats) != len(r_lats):
        raise ValueError("Latency arrays must have the same length")
    if len(m_lats) == 0 or np.all(m_lats == r_lats):
        return 1.0

    try:
        res = wilcoxon(m_lats, r_lats)
        return float(res.pvalue)
    except Exception:
        return 1.0


def holm_adjust(
    p_values: list[float] | dict[str, float]
) -> list[float] | dict[str, float]:
    """Holm-Bonferroni step-down adjustment."""
    is_dict = isinstance(p_values, dict)
    if is_dict:
        keys = list(p_values.keys())
        raw_p = [p_values[k] for k in keys]
    else:
        raw_p = list(p_values)

    m = len(raw_p)
    if m == 0:
        return {} if is_dict else []

    sorted_indices = sorted(range(m), key=lambda i: raw_p[i])
    adj_sorted = [0.0] * m
    cum_max = 0.0
    for rank, idx in enumerate(sorted_indices):
        val = (m - rank) * raw_p[idx]
        cum_max = max(cum_max, val)
        adj_sorted[rank] = min(1.0, cum_max)

    adjusted = [0.0] * m
    for orig_idx, adj_val in zip(sorted_indices, adj_sorted):
        adjusted[orig_idx] = adj_val

    if is_dict:
        return {keys[i]: adjusted[i] for i in range(m)}
    return adjusted


def compute_within_case_latency_variability(
    rows: list[dict[str, Any]],
    subset_ids: set[str] | list[str] | None = None,
) -> dict[str, Any]:
    """Compute within-case latency variability (CV over repeats 0-2 on the subset)."""
    case_lats: dict[str, dict[int, float]] = {}
    for r in rows:
        cid = str(r.get("case_id", ""))
        rep = r.get("repeat")
        lat = r.get("latency_s")
        if subset_ids is not None and cid not in subset_ids:
            continue
        if rep in (0, 1, 2) and lat is not None:
            if cid not in case_lats:
                case_lats[cid] = {}
            case_lats[cid][int(rep)] = float(lat)

    cv_list: list[float] = []
    for cid, reps in case_lats.items():
        if len(reps) >= 2:
            vals = list(reps.values())
            m = float(np.mean(vals))
            s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            cv = s / m if m > 0 else 0.0
            cv_list.append(cv)

    mean_cv = float(np.mean(cv_list)) if cv_list else 0.0
    median_cv = float(np.median(cv_list)) if cv_list else 0.0

    return {
        "cv": mean_cv,
        "mean_cv": mean_cv,
        "median_cv": median_cv,
        "n_cases": len(cv_list),
        "case_cvs": cv_list,
    }


def aggregate_rq1b_metrics(
    rows: list[dict[str, Any]],
    seed: int = 20260924,
    n_resamples: int = 10000,
) -> dict[str, Any]:
    """Aggregate RQ1b metrics: request-level EM with message-cluster bootstrap, secondary message EM."""
    r0_rows = [r for r in rows if r.get("repeat", 0) == 0]
    n_messages = len(r0_rows)
    if n_messages == 0:
        return {"n_messages": 0, "n_requests": 0}

    msg_request_ems: list[list[bool]] = []
    for r in r0_rows:
        req_em = r.get("request_em")
        if req_em is not None and isinstance(req_em, list):
            msg_request_ems.append([bool(x) for x in req_em])
        else:
            msg_request_ems.append([bool(r.get("em", False))])

    total_requests = sum(len(x) for x in msg_request_ems)
    total_correct = sum(sum(x) for x in msg_request_ems)
    request_level_em = float(total_correct / total_requests) if total_requests > 0 else 0.0

    # Message-cluster bootstrap for request-level EM
    rng = np.random.default_rng(seed)
    boot_indices = rng.integers(0, n_messages, size=(n_resamples, n_messages))

    msg_sums = np.array([sum(x) for x in msg_request_ems], dtype=float)
    msg_counts = np.array([len(x) for x in msg_request_ems], dtype=float)

    resample_sums = np.sum(msg_sums[boot_indices], axis=1)
    resample_counts = np.sum(msg_counts[boot_indices], axis=1)
    boot_req_ems = np.where(resample_counts > 0, resample_sums / resample_counts, 0.0)

    ci_low = float(np.percentile(boot_req_ems, 2.5))
    ci_high = float(np.percentile(boot_req_ems, 97.5))
    ci_low = min(ci_low, request_level_em)
    ci_high = max(ci_high, request_level_em)
    request_level_em_ci95 = (ci_low, ci_high)

    msg_ems = [bool(r.get("em", False)) for r in r0_rows]
    message_em = float(np.mean(msg_ems))
    message_em_ci95 = bootstrap_ci(msg_ems, seed=seed, n_resamples=n_resamples)

    raw_lats = [float(r["latency_s"]) for r in r0_rows if r.get("latency_s") is not None]
    per_req_lats = []
    for r in r0_rows:
        lat = r.get("latency_s")
        if lat is not None:
            k = len(r.get("request_em", [])) or 1
            per_req_lats.append(float(lat) / k)

    p50_raw = float(np.percentile(raw_lats, 50)) if raw_lats else 0.0
    p95_raw = float(np.percentile(raw_lats, 95)) if raw_lats else 0.0
    p99_raw = float(np.percentile(raw_lats, 99)) if raw_lats else 0.0

    p50_per_req = float(np.percentile(per_req_lats, 50)) if per_req_lats else 0.0
    p95_per_req = float(np.percentile(per_req_lats, 95)) if per_req_lats else 0.0
    p99_per_req = float(np.percentile(per_req_lats, 99)) if per_req_lats else 0.0

    return {
        "n_messages": n_messages,
        "n_requests": total_requests,
        "request_level_em": request_level_em,
        "request_level_em_ci95": request_level_em_ci95,
        "message_em": message_em,
        "message_em_ci95": message_em_ci95,
        "latency_raw_p50": p50_raw,
        "latency_raw_p95": p95_raw,
        "latency_raw_p99": p99_raw,
        "per_request_latency_p50": p50_per_req,
        "per_request_latency_p95": p95_per_req,
        "per_request_latency_p99": p99_per_req,
    }


def compute_growth_ratio(
    latencies_level: list[float] | np.ndarray,
    latencies_ref_level: list[float] | np.ndarray,
    seed: int = 20260924,
    n_resamples: int = 10000,
) -> tuple[float, tuple[float, float]]:
    """Compute latency growth ratio G(level) = p50(level) / p50(ref_level) with paired bootstrap CI."""
    l1 = np.asarray(latencies_level, dtype=float)
    l0 = np.asarray(latencies_ref_level, dtype=float)
    if len(l1) != len(l0):
        raise ValueError("Latencies arrays must have the same length for paired growth ratio")
    n = len(l1)
    if n == 0:
        return 0.0, (0.0, 0.0)

    p50_1 = float(np.percentile(l1, 50))
    p50_0 = float(np.percentile(l0, 50))
    point = p50_1 / p50_0 if p50_0 > 0 else 0.0

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_resamples, n))
    resample_p50_1 = np.percentile(l1[indices], 50, axis=1)
    resample_p50_0 = np.percentile(l0[indices], 50, axis=1)

    ratios = np.where(resample_p50_0 > 0, resample_p50_1 / resample_p50_0, 0.0)
    ci_low = float(np.percentile(ratios, 2.5))
    ci_high = float(np.percentile(ratios, 97.5))
    ci_low = min(ci_low, point)
    ci_high = max(ci_high, point)
    return point, (ci_low, ci_high)


def _stratified_bootstrap_indices(
    strata: list[Any] | np.ndarray, n_resamples: int, rng: np.random.Generator
) -> np.ndarray:
    """Bootstrap index matrix (n_resamples, n) resampling cases independently within each stratum."""
    labels = np.asarray(strata)
    blocks = []
    for label in sorted(set(labels.tolist()), key=str):
        members = np.flatnonzero(labels == label)
        draws = rng.integers(0, len(members), size=(n_resamples, len(members)))
        blocks.append(members[draws])
    return np.concatenate(blocks, axis=1)


def compute_ratio_of_ratios(
    lats_m1_level: list[float] | np.ndarray,
    lats_m1_ref: list[float] | np.ndarray,
    lats_m2_level: list[float] | np.ndarray,
    lats_m2_ref: list[float] | np.ndarray,
    seed: int = 20260924,
    n_resamples: int = 10000,
    strata: list[Any] | np.ndarray | None = None,
) -> tuple[float, tuple[float, float]]:
    """Compute ratio-of-ratios G_m1 / G_m2 with a case-paired bootstrap CI.

    All four arrays are aligned by case and resampled with the same indices. With `strata`
    (e.g. the D level of each case for H4), cases are resampled independently within each
    stratum; the point estimate stays pooled over strata.
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
        return 0.0, (0.0, 0.0)

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
    return point, (ci_low, ci_high)


def invert_bootstrap_pvalue(
    bootstrap_samples: list[float] | np.ndarray,
    null_value: float = 0.0,
) -> float:
    """Invert bootstrap distribution into a two-sided p-value against null_value."""
    boot = np.asarray(bootstrap_samples, dtype=float)
    if len(boot) == 0:
        return 1.0
    p_left = float(np.mean(boot <= null_value))
    p_right = float(np.mean(boot >= null_value))
    p_val = min(1.0, 2.0 * min(p_left, p_right))
    return float(p_val)


def _prf(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precision/recall/F1 arrays; NaN where undefined."""
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
        recall = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), np.nan)
    return precision, recall, f1


def _nan_ci(samples: np.ndarray, point: float | None) -> tuple[float, float] | None:
    """95% percentile CI over finite replicates, widened to contain the point estimate."""
    finite = samples[np.isfinite(samples)]
    if len(finite) == 0:
        return None
    lo = float(np.percentile(finite, 2.5))
    hi = float(np.percentile(finite, 97.5))
    if point is not None and np.isfinite(point):
        lo, hi = min(lo, point), max(hi, point)
    return (lo, hi)


def compute_catalog_metrics(
    rows: list[dict[str, Any]],
    cases_map: dict[str, Case] | None = None,
    seed: int = 20260924,
    n_resamples: int = 10000,
) -> dict[str, Any]:
    """Catalog metrics with 95% case-bootstrap CIs.

    Seen vs unseen top-1 service accuracy, unsupported precision/recall/F1, and the seen - unseen
    gap. Each bootstrap replicate resamples cases (jointly across seen and unseen) and recomputes
    every metric, including the gap; replicates where a metric is undefined are skipped.
    `seen_unseen_gap_ci_upper` is the upper CI bound used by the H5 rule.
    """
    n = len(rows)
    seen_flag = np.zeros(n, dtype=bool)
    unseen_flag = np.zeros(n, dtype=bool)
    svc_ok = np.zeros(n, dtype=float)
    tp_c = np.zeros(n, dtype=float)
    fp_c = np.zeros(n, dtype=float)
    fn_c = np.zeros(n, dtype=float)
    tn = 0

    for i, r in enumerate(rows):
        case_id = str(r["case_id"])
        meta = r.get("meta", {})
        if cases_map and case_id in cases_map:
            meta = cases_map[case_id].meta or meta
        is_seen = meta.get("seen")

        correct_dict = r.get("correct", {})
        service_correct = correct_dict.get("service_type")

        labels = r.get("labels", [])
        truth = []
        if cases_map and case_id in cases_map:
            truth = cases_map[case_id].truth

        # Seen/unseen buckets hold supported targets only: an unsupported target (truth "unsupported") or
        # seen=None is in neither; it is scored in the unsupported bucket below.
        unsupported_target = any(t.get("service_type") == "unsupported" for t in truth)
        if is_seen is True and service_correct is not None and not unsupported_target:
            seen_flag[i] = True
            svc_ok[i] = float(bool(service_correct))
        elif is_seen is False and service_correct is not None and not unsupported_target:
            unseen_flag[i] = True
            svc_ok[i] = float(bool(service_correct))

        for pred_req, truth_req in zip(labels, truth):
            pred_svc = pred_req.get("service_type")
            truth_svc = truth_req.get("service_type")
            truth_unsupp = truth_svc == "unsupported"
            pred_unsupp = pred_svc == "unsupported"

            if truth_unsupp and pred_unsupp:
                tp_c[i] += 1
            elif not truth_unsupp and pred_unsupp:
                fp_c[i] += 1
            elif truth_unsupp and not pred_unsupp:
                fn_c[i] += 1
            else:
                tn += 1

    tp, fp, fn = int(tp_c.sum()), int(fp_c.sum()), int(fn_c.sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    seen_n = int(seen_flag.sum())
    unseen_n = int(unseen_flag.sum())
    seen_acc = float(svc_ok[seen_flag].mean()) if seen_n else None
    unseen_acc = float(svc_ok[unseen_flag].mean()) if unseen_n else None
    gap = seen_acc - unseen_acc if (seen_acc is not None and unseen_acc is not None) else None

    out: dict[str, Any] = {
        "seen_accuracy": seen_acc,
        "seen_n": seen_n,
        "unseen_accuracy": unseen_acc,
        "unseen_n": unseen_n,
        "unsupported_tp": tp,
        "unsupported_fp": fp,
        "unsupported_fn": fn,
        "unsupported_tn": tn,
        "unsupported_precision": float(precision),
        "unsupported_recall": float(recall),
        "unsupported_f1": float(f1),
        "seen_unseen_gap": gap,
        "bootstrap": {"seed": seed, "n_resamples": n_resamples, "unit": "case"},
    }

    ci_keys = ("seen_accuracy", "unseen_accuracy", "unsupported_precision",
               "unsupported_recall", "unsupported_f1", "seen_unseen_gap")
    if n == 0:
        out.update({f"{k}_ci95": None for k in ci_keys})
        out["seen_unseen_gap_ci_upper"] = None
        return out

    idx = np.random.default_rng(seed).integers(0, n, size=(n_resamples, n))
    seen_cnt = seen_flag[idx].sum(axis=1)
    unseen_cnt = unseen_flag[idx].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        boot_seen = np.where(seen_cnt > 0, (svc_ok * seen_flag)[idx].sum(axis=1) / seen_cnt, np.nan)
        boot_unseen = np.where(unseen_cnt > 0, (svc_ok * unseen_flag)[idx].sum(axis=1) / unseen_cnt, np.nan)
    boot_p, boot_r, boot_f1 = _prf(tp_c[idx].sum(axis=1), fp_c[idx].sum(axis=1), fn_c[idx].sum(axis=1))

    samples = {
        "seen_accuracy": boot_seen,
        "unseen_accuracy": boot_unseen,
        "unsupported_precision": boot_p,
        "unsupported_recall": boot_r,
        "unsupported_f1": boot_f1,
        "seen_unseen_gap": boot_seen - boot_unseen,
    }
    for key in ci_keys:
        out[f"{key}_ci95"] = _nan_ci(samples[key], out[key])
    gap_ci = out["seen_unseen_gap_ci95"]
    out["seen_unseen_gap_ci_upper"] = gap_ci[1] if gap_ci is not None else None
    return out


def load_ledger_rows(
    ledger_path: str | Path, manifest: dict[str, dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """Read ledger rows for scoring; Laya rows get input_tokens / truncated from the offline sidecar.

    The sidecar `<ledger>.laya_tokens.jsonl` is joined by (condition, case_id). Without it (or without a
    record for a row) the Laya token counts stay None, with a warning; they are never read as 0.
    """
    ledger_path = Path(ledger_path)
    with open(ledger_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    manifest = load_manifest() if manifest is None else manifest
    laya_names = {
        name for name, entry in manifest.items() if str(entry.get("model", "")).startswith(LAYA_MODEL_PREFIX)
    }
    laya_rows = [r for r in rows if r.get("model") in laya_names]
    if not laya_rows:
        return rows

    sidecar = Path(f"{ledger_path}.laya_tokens.jsonl")
    counts: dict[tuple[str, str], dict[str, Any]] = {}
    if sidecar.exists():
        with open(sidecar, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    counts[(str(rec["condition"]), str(rec["case_id"]))] = rec
    else:
        warnings.warn(
            f"{sidecar.name} not found: Laya token metrics are None (run scripts/eb_laya_tokens.py)",
            stacklevel=2,
        )
    missing = 0
    for r in laya_rows:
        rec = counts.get((str(r.get("condition")), str(r.get("case_id"))))
        r["input_tokens"] = rec["input_tokens"] if rec else None
        r["truncated"] = rec["truncated"] if rec else None
        missing += rec is None
    if counts and missing:
        warnings.warn(f"{missing} Laya rows have no record in {sidecar.name}: their token metrics are None",
                      stacklevel=2)
    return rows


def aggregate_metrics(
    rows: list[dict[str, Any]],
    cases_map: dict[str, Case] | None = None,
    reference_rows: list[dict[str, Any]] | None = None,
    seed: int = 20260924,
) -> dict[str, Any]:
    """Aggregate metrics across evaluation rows for a (model, condition).

    Primary aggregation uses repeat 0 only.
    Raises KeyError if any row lacks persisted scoring outputs.
    """
    # Primary aggregation must use repeat 0 only
    r0_rows = [r for r in rows if r.get("repeat", 0) == 0]
    n = len(r0_rows)
    if n == 0:
        return {"n": 0}

    # Validate required scoring fields in every row (no silent default)
    required_scoring_fields = (
        "unsafe_locality",
        "spurious_count",
        "missed_count",
        "unspecified_truth_count",
        "specified_truth_count",
    )
    for r in r0_rows:
        for f in required_scoring_fields:
            if f not in r:
                raise KeyError(f"Row for case {r.get('case_id')} missing required scoring field: '{f}'")

    em_list = [bool(r.get("em", False)) for r in r0_rows]
    em_rate = float(np.mean(em_list))
    em_ci = bootstrap_ci(em_list, seed=seed, n_resamples=10000)

    # Per-field accuracies and macro accuracy
    all_fields: set[str] = set()
    for r in r0_rows:
        all_fields.update(r.get("correct", {}).keys())
    field_accuracies = {}
    for f in sorted(all_fields):
        vals = [bool(r.get("correct", {}).get(f, False)) for r in r0_rows]
        field_accuracies[f] = float(np.mean(vals))

    case_field_accs = [
        float(np.mean([bool(r.get("correct", {}).get(f, False)) for f in all_fields]))
        if all_fields else 0.0
        for r in r0_rows
    ]
    macro_field_accuracy = float(np.mean(case_field_accs)) if case_field_accs else 0.0
    macro_field_accuracy_ci = bootstrap_ci(case_field_accs, seed=seed, n_resamples=10000)

    valid_list = [bool(r.get("valid", True)) for r in r0_rows]
    valid_rate = float(np.mean(valid_list))
    valid_ci = bootstrap_ci(valid_list, seed=seed, n_resamples=10000)

    unsafe_list = [bool(r.get("unsafe_locality", False)) for r in r0_rows]
    unsafe_rate = float(np.mean(unsafe_list))
    unsafe_ci = bootstrap_ci(unsafe_list, seed=seed, n_resamples=10000)

    # Spurious and missed rates
    spurious_counts = [int(r["spurious_count"]) for r in r0_rows]
    unspec_truth_counts = [int(r["unspecified_truth_count"]) for r in r0_rows]
    total_spurious = sum(spurious_counts)
    total_unspec_truth = sum(unspec_truth_counts)
    spurious_rate = (
        total_spurious / total_unspec_truth if total_unspec_truth > 0 else 0.0
    )
    spurious_ci = _bootstrap_ratio_ci(spurious_counts, unspec_truth_counts, seed=seed, n_resamples=10000)

    missed_counts = [int(r["missed_count"]) for r in r0_rows]
    spec_truth_counts = [int(r["specified_truth_count"]) for r in r0_rows]
    total_missed = sum(missed_counts)
    total_spec_truth = sum(spec_truth_counts)
    missed_rate = total_missed / total_spec_truth if total_spec_truth > 0 else 0.0
    missed_ci = _bootstrap_ratio_ci(missed_counts, spec_truth_counts, seed=seed, n_resamples=10000)

    # Latencies
    lats = [float(r["latency_s"]) for r in r0_rows if r.get("latency_s") is not None]
    if lats:
        p50, p95, p99 = np.percentile(lats, [50, 95, 99], method="linear")
        p50_ci = bootstrap_ci(
            lats,
            seed=seed,
            n_resamples=10000,
            stat_fn=lambda a, **kw: np.percentile(a, 50, **kw),
        )
        p95_ci = bootstrap_ci(
            lats,
            seed=seed,
            n_resamples=10000,
            stat_fn=lambda a, **kw: np.percentile(a, 95, **kw),
        )
    else:
        p50 = p95 = p99 = 0.0
        p50_ci = (0.0, 0.0)
        p95_ci = (0.0, 0.0)

    # Tokens and cost: null = not reported. Totals and means are over the rows that reported the value
    # (None when none did); n_* is the number of rows used.
    def _known(key: str, cast: Any) -> list[Any]:
        return [cast(r[key]) for r in r0_rows if r.get(key) is not None]

    in_tokens = _known("input_tokens", int)
    out_tokens = _known("output_tokens", int)
    reason_tokens = _known("reasoning_tokens", int)

    total_in = sum(in_tokens) if in_tokens else None
    total_out = sum(out_tokens) if out_tokens else None
    total_reason = sum(reason_tokens) if reason_tokens else None

    # Cost
    cost_rows = [r for r in r0_rows if r.get("cost_usd") is not None]
    total_cost_usd = sum(float(r["cost_usd"]) for r in cost_rows) if cost_rows else None
    em_correct_count = sum(bool(r.get("em", False)) for r in cost_rows)
    cost_per_1000_correct = (
        (total_cost_usd / em_correct_count * 1000.0)
        if total_cost_usd is not None and em_correct_count > 0
        else float("nan")
    )

    out: dict[str, Any] = {
        "n": n,
        "em": em_rate,
        "em_ci95": em_ci,
        "macro_field_accuracy": macro_field_accuracy,
        "macro_field_accuracy_ci95": macro_field_accuracy_ci,
        "field_accuracies": field_accuracies,
        "valid_rate": valid_rate,
        "valid_rate_ci95": valid_ci,
        "unsafe_rate": unsafe_rate,
        "unsafe_rate_ci95": unsafe_ci,
        "spurious_rate": float(spurious_rate),
        "spurious_rate_ci95": spurious_ci,
        "missed_rate": float(missed_rate),
        "missed_rate_ci95": missed_ci,
        "latency_p50": float(p50),
        "latency_p50_ci95": p50_ci,
        "latency_p95": float(p95),
        "latency_p95_ci95": p95_ci,
        "latency_p99": float(p99),
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "total_reasoning_tokens": total_reason,
        "mean_input_tokens": total_in / len(in_tokens) if in_tokens else None,
        "mean_output_tokens": total_out / len(out_tokens) if out_tokens else None,
        "mean_reasoning_tokens": total_reason / len(reason_tokens) if reason_tokens else None,
        "n_input_tokens": len(in_tokens),
        "n_output_tokens": len(out_tokens),
        "n_reasoning_tokens": len(reason_tokens),
        "total_cost_usd": total_cost_usd,
        "n_cost": len(cost_rows),
        "cost_per_1000_correct": cost_per_1000_correct,
    }

    # Catalog metrics
    catalog_metrics = compute_catalog_metrics(r0_rows, cases_map=cases_map, seed=seed)
    out["catalog"] = catalog_metrics

    # Paired metrics vs reference model if provided
    if reference_rows:
        ref_r0 = [r for r in reference_rows if r.get("repeat", 0) == 0]
        ref_by_key = {str(r["case_id"]): r for r in ref_r0}

        paired_m_em = []
        paired_r_em = []
        paired_m_unsafe = []
        paired_r_unsafe = []
        paired_m_lat = []
        paired_r_lat = []

        for r in r0_rows:
            cid = str(r["case_id"])
            if cid in ref_by_key:
                ref_r = ref_by_key[cid]
                paired_m_em.append(bool(r.get("em", False)))
                paired_r_em.append(bool(ref_r.get("em", False)))
                paired_m_unsafe.append(bool(r.get("unsafe_locality", False)))
                paired_r_unsafe.append(bool(ref_r.get("unsafe_locality", False)))
                # Latency pairs exclude rows with no measured latency (None); quality pairs keep them.
                if r.get("latency_s") is not None and ref_r.get("latency_s") is not None:
                    paired_m_lat.append(float(r["latency_s"]))
                    paired_r_lat.append(float(ref_r["latency_s"]))

        if paired_m_em:
            diffs_em = np.asarray(paired_m_em, dtype=float) - np.asarray(
                paired_r_em, dtype=float
            )
            paired_em_diff = float(np.mean(diffs_em))
            paired_em_ci = bootstrap_ci(diffs_em, seed=seed, n_resamples=10000)
            mcnemar_em_p = exact_mcnemar_test(paired_m_em, paired_r_em)

            diffs_unsafe = np.asarray(paired_m_unsafe, dtype=float) - np.asarray(
                paired_r_unsafe, dtype=float
            )
            paired_unsafe_diff = float(np.mean(diffs_unsafe))
            paired_unsafe_ci = bootstrap_ci(diffs_unsafe, seed=seed, n_resamples=10000)
            mcnemar_unsafe_p = exact_mcnemar_test(paired_m_unsafe, paired_r_unsafe)

            diffs_lat = np.asarray(paired_m_lat, dtype=float) - np.asarray(
                paired_r_lat, dtype=float
            )
            latency_diff_median = float(np.median(diffs_lat)) if len(diffs_lat) else float("nan")
            latency_diff_median_ci = bootstrap_ci(
                diffs_lat,
                seed=seed,
                n_resamples=10000,
                stat_fn=lambda a, **kw: np.median(a, **kw),
            )
            wilcoxon_lat_p = paired_wilcoxon_test(paired_m_lat, paired_r_lat)

            out["paired_vs_ref"] = {
                "paired_n": len(paired_m_em),
                "paired_latency_n": len(paired_m_lat),
                "em_diff": paired_em_diff,
                "em_diff_ci95": paired_em_ci,
                "mcnemar_p": mcnemar_em_p,
                "unsafe_diff": paired_unsafe_diff,
                "unsafe_diff_ci95": paired_unsafe_ci,
                "unsafe_mcnemar_p": mcnemar_unsafe_p,
                "latency_diff_median": latency_diff_median,
                "latency_diff_median_ci95": latency_diff_median_ci,
                "wilcoxon_latency_p": wilcoxon_lat_p,
            }

    return out
