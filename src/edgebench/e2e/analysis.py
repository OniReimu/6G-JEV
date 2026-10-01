"""EXP-2026-002 RQ5 analysis (H6-H8): block bootstrap over contiguous arrival blocks, paired across interpreters.

Frozen plan: the pre-registered analysis plan of the edge orchestration study (with deviations D-1..D-3).

Completion metrics per arrival (denominator = supported arrivals):
- Part A operational: served, predicted service == truth service, locality (truth remote_allowed or run on origin),
  minimum tier (truth high -> predicted high), mapped priority (urgent -> 0, else 1) equal to truth's, finish <= deadline.
- Part A strict: operational and exact equality of all four fields (== simulator status "success"; cross-checked).
- Part B operational: score.correct_completion. Part B strict: operational and score.strict_semantic.
Primary metric used in G, shared successes and H6-H8: Part A strict (simulator success), Part B score.correct_completion.
"""
from __future__ import annotations

import csv
from io import StringIO
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.edgebench.analysis import HOSTED_MODELS, REFERENCE_MODEL, SELFHOSTED_MODELS, THE_SIX_MODELS
from src.edgebench.e2e.sim import simulate
from src.edgebench.e2e.traces import CELLS_A, CONDITIONS_B
from src.edgebench.scoring import holm_adjust, invert_bootstrap_pvalue

SEED = 20260924
N_RESAMPLES = 10_000
BLOCK_SIZE = 30
EXPECTED_N = {"A": 300, "B": 240}
MODELS = THE_SIX_MODELS + [REFERENCE_MODEL]
HOSTED_ANCHOR = "Jev-1.13.0"
SELF_ANCHOR = "SemIf-Qwen3.5-4B"
HOSTED_CONTRASTS = [m for m in HOSTED_MODELS if m != HOSTED_ANCHOR]
SELF_CONTRAST = REFERENCE_MODEL  # H6 self-hosted analogue: SemIf vs Qwen3.5-4B-JSON
CELLS = {"A": list(CELLS_A), "B": list(CONDITIONS_B)}
LOAD_CELLS = [("load_1", 1.0), ("load_2", 2.0), ("load_4", 4.0), ("load_8", 8.0), ("load_16", 16.0)]
CACHE_OFF = {
    "A": [c for c, cfg in CELLS_A.items() if not cfg["cache"]],
    "B": [c for c, cfg in CONDITIONS_B.items() if not cfg["cache"]],
}
H8_PAIRS = {
    "A": [("reuse_repeated_off", "reuse_repeated_on")],
    "B": [("steady_repeated_off", "steady_repeated_on"), ("bursty_repeated_off", "bursty_repeated_on")],
}
TOPOLOGY_N = [5, 10, 20, 40]
MIN_SHARED = 30
MIN_ELIGIBLE = 5
H7_SHARE = 0.8
ALPHA = 0.05

STATUS_A = [
    "success", "wrong_or_late", "queue_expired", "queue_overflow", "decision_late", "decision_timeout",
    "invalid_decision", "unsupported", "no_feasible_node",
]
STATUS_B = [
    "ocr_returned", "ocr_error", "undeployed", "unsupported", "decision_late", "queue_expired", "queue_overflow",
    "invalid_decision", "invalid_labels", "no_feasible_node", "service_queue_expired", "service_aborted",
    "service_transport_or_integrity_error", "aborted", "aborted_before_dispatch",
]
STATUS_ALL = STATUS_A + [s for s in STATUS_B if s not in STATUS_A]
SERVED_A = ("success", "wrong_or_late")


# ---------------------------------------------------------------------------
# Per-arrival scoring
# ---------------------------------------------------------------------------

def deployment(model: str) -> str:
    if model in HOSTED_MODELS:
        return "hosted"
    if model in SELFHOSTED_MODELS:
        return "self-hosted"
    return "control"


def anchor_for(model: str) -> str | None:
    return {"hosted": HOSTED_ANCHOR, "self-hosted": SELF_ANCHOR}.get(deployment(model))


def _truth(row: dict[str, Any]) -> dict[str, Any]:
    t = row.get("truth")
    return t[0] if isinstance(t, list) else t


def mapped_priority(urgency: str | None) -> int:
    return 0 if urgency == "urgent" else 1


def terminal_status(part: str, row: dict[str, Any]) -> str:
    """Terminal status; Part B 'decision_recorded' with null prediction is an invalid interpreter output."""
    status = row["status"]
    if part == "B" and status == "decision_recorded":
        if row.get("predicted") is not None:
            raise ValueError(f"decision_recorded with a prediction is not terminal (id {row.get('id')})")
        status = "invalid_decision"
    if status not in (STATUS_A if part == "A" else STATUS_B):
        raise ValueError(f"unknown Part {part} status {status!r} (id {row.get('id')})")
    return status


def supported(part: str, row: dict[str, Any]) -> bool:
    service = _truth(row).get("service_type")
    return service == "ocr" if part == "B" else service != "unsupported"


def operational_a(row: dict[str, Any]) -> bool:
    """Part A operational completion (service, locality, minimum tier, mapped priority, finish <= deadline)."""
    truth, pred = _truth(row), row.get("predicted")
    if row["status"] not in SERVED_A or not isinstance(pred, dict):
        return False
    return bool(
        truth.get("service_type") != "unsupported"
        and pred.get("service_type") == truth.get("service_type")
        and (truth.get("locality") == "remote_allowed" or row.get("node") == row.get("origin"))
        and (truth.get("quality_floor") != "high" or pred.get("quality_floor") == "high")
        and row.get("priority") == mapped_priority(truth.get("urgency"))
        and row["terminal"] <= row["deadline"]
    )


def strict_a(row: dict[str, Any]) -> bool:
    """Part A strict completion: operational plus exact equality of all four fields; must equal simulator success."""
    strict = operational_a(row) and row.get("predicted") == _truth(row)
    if strict != (row["status"] == "success"):
        raise ValueError(f"strict completion disagrees with simulator status for id {row.get('id')}")
    return strict


def false_admission_b(row: dict[str, Any]) -> bool:
    """Part B false admission: non-OCR truth that was actually dispatched to an OCR worker."""
    return _truth(row).get("service_type") != "ocr" and row.get("service_dispatch") is not None


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

def block_index(n: int, block_size: int = BLOCK_SIZE) -> np.ndarray:
    """Block of each arrival position (positions in arrival order): contiguous runs of block_size."""
    return np.arange(n) // block_size


def build_arm(part: str, cell: str, model: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = sorted(rows, key=lambda r: (r["arrival"], r["id"]))
    status = [terminal_status(part, r) for r in rows]
    sup = np.array([supported(part, r) for r in rows], dtype=bool)
    if part == "A":
        op = np.array([operational_a(r) for r in rows], dtype=bool)
        st = np.array([strict_a(r) for r in rows], dtype=bool)
        primary = st
    else:
        op = np.array([bool(r["score"]["correct_completion"]) for r in rows], dtype=bool)
        st = op & np.array([bool(r["score"]["strict_semantic"]) for r in rows], dtype=bool)
        primary = op
    return dict(
        part=part, cell=cell, model=model, rows=rows, status=status,
        ids=tuple(r["id"] for r in rows),
        supported=sup, operational=op & sup, strict=st & sup, primary=primary & sup,
        T=np.array([r["T"] for r in rows], dtype=float),
        block=block_index(len(rows)),
    )


def block_draws(n_blocks: int, n_resamples: int = N_RESAMPLES, seed: int = SEED) -> np.ndarray:
    """Multiplicity matrix (n_resamples x n_blocks) of blocks drawn with replacement."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_blocks, size=(n_resamples, n_blocks))
    counts = np.zeros((n_resamples, n_blocks), dtype=np.int64)
    np.add.at(counts, (np.repeat(np.arange(n_resamples), n_blocks), idx.ravel()), 1)
    return counts


def boot_rate(flags: np.ndarray, denom: np.ndarray, block: np.ndarray, counts: np.ndarray) -> np.ndarray:
    k = counts.shape[1]
    num = np.bincount(block, weights=(flags & denom).astype(float), minlength=k)
    den = np.bincount(block, weights=denom.astype(float), minlength=k)
    with np.errstate(divide="ignore", invalid="ignore"):  # elementwise sums: Accelerate matmul raises spurious FP flags
        return (counts * num).sum(axis=1) / (counts * den).sum(axis=1)


def rate(flags: np.ndarray, denom: np.ndarray) -> float | None:
    d = int(denom.sum())
    return float((flags & denom).sum() / d) if d else None


def boot_percentile(values: np.ndarray, block: np.ndarray, counts: np.ndarray, q: float) -> np.ndarray:
    """np.percentile (linear) of each block resample of `values`, computed from block multiplicities."""
    out = np.full(counts.shape[0], np.nan)
    if len(values) == 0:
        return out
    order = np.argsort(values, kind="stable")
    v, b = values[order], block[order]
    cum = np.cumsum(counts[:, b], axis=1)
    n = cum[:, -1]
    ok = n > 0
    h = np.where(ok, (n - 1) * q / 100.0, 0.0)
    lo = np.floor(h)
    hi = np.minimum(lo + 1, np.maximum(n - 1, 0))
    i_lo = np.minimum((cum <= lo[:, None]).sum(axis=1), len(v) - 1)
    i_hi = np.minimum((cum <= hi[:, None]).sum(axis=1), len(v) - 1)
    res = v[i_lo] + (h - lo) * (v[i_hi] - v[i_lo])
    out[ok] = res[ok]
    return out


def percentile(values: np.ndarray, q: float) -> float | None:
    return float(np.percentile(values, q)) if len(values) else None


def ci(samples: np.ndarray, point: float | None) -> tuple[float | None, float | None]:
    finite = samples[np.isfinite(samples)]
    if point is None or len(finite) == 0:
        return None, None
    return min(float(np.percentile(finite, 2.5)), point), max(float(np.percentile(finite, 97.5)), point)


def pvalue(samples: np.ndarray) -> float | None:
    finite = samples[np.isfinite(samples)]
    if len(finite) == 0:
        return None
    return max(invert_bootstrap_pvalue(finite, null_value=0.0), 1.0 / (len(finite) + 1))


def draws_for(arm: dict[str, Any], cache: dict[int, np.ndarray]) -> np.ndarray:
    """Same block draws for every arm with the same block count (all models of a cell; cells share positions)."""
    k = int(arm["block"].max()) + 1 if len(arm["block"]) else 1
    if k not in cache:
        cache[k] = block_draws(k)
    return cache[k]


def pair_stats(arm_l: dict[str, Any], arm_a: dict[str, Any], cache: dict[int, np.ndarray]) -> dict[str, Any]:
    """Paired statistics of interpreter l against anchor a on one trace (same block draws for both)."""
    if arm_l["ids"] != arm_a["ids"]:
        raise ValueError(f"arms are not on the same trace: {arm_l['cell']} {arm_l['model']} vs {arm_a['model']}")
    counts = draws_for(arm_a, cache)
    blk, sup = arm_a["block"], arm_a["supported"]
    out: dict[str, Any] = {}
    for metric in ("primary", "operational"):
        pt_a, pt_l = rate(arm_a[metric], sup), rate(arm_l[metric], sup)
        out[f"G_{metric}"] = None if pt_a is None or pt_l is None else pt_a - pt_l
        out[f"G_{metric}_boot"] = boot_rate(arm_a[metric], sup, blk, counts) - boot_rate(arm_l[metric], sup, blk, counts)
    shared = arm_a["primary"] & arm_l["primary"]
    out["n_shared"] = int(shared.sum())
    for q in (95, 50):
        pl, pa = percentile(arm_l["T"][shared], q), percentile(arm_a["T"][shared], q)
        out[f"dT{q}"] = None if pl is None else pl - pa
        out[f"dT{q}_boot"] = (
            boot_percentile(arm_l["T"][shared], blk[shared], counts, q)
            - boot_percentile(arm_a["T"][shared], blk[shared], counts, q)
        )
    return out


# ---------------------------------------------------------------------------
# Loading and coverage
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def check_arm_dir(arm_dir: Path, expected_n: int) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Return (outcome rows, None) for a complete arm, else (None, reason). Never reads an arm without integrity.json."""
    integrity = arm_dir / "integrity.json"
    if not integrity.exists():
        return None, "no integrity.json (missing or still running)"
    integ = json.loads(integrity.read_text(encoding="utf-8"))
    if not integ.get("complete"):
        return None, "integrity.complete is false"
    if integ.get("arrivals") != expected_n:
        return None, f"integrity arrivals {integ.get('arrivals')} != {expected_n}"
    rows = _read_jsonl(arm_dir / "outcomes.jsonl")
    if len(rows) != expected_n or len({r["id"] for r in rows}) != len(rows):
        return None, f"outcomes rows {len(rows)} (unique ids {len({r['id'] for r in rows})}) != {expected_n}"
    return rows, None


def load_part(root: Path, part: str) -> tuple[dict[tuple[str, str], dict[str, Any]], list[dict[str, str]], dict]:
    """Load all expected arms of a part plus any control arms. Returns (arms, gaps, controls)."""
    arms: dict[tuple[str, str], dict[str, Any]] = {}
    gaps: list[dict[str, str]] = []
    for cell in CELLS[part]:
        for model in MODELS:
            rows, reason = check_arm_dir(root / cell / "seed_1" / model, EXPECTED_N[part])
            if rows is None:
                gaps.append(dict(part=part, cell=cell, model=model, reason=reason))
            else:
                arms[(cell, model)] = build_arm(part, cell, model, rows)
    controls: dict[tuple[str, str], dict[str, Any]] = {}
    for cell in CELLS[part]:
        seed_dir = root / cell / "seed_1"
        if not seed_dir.is_dir():
            continue
        for d in sorted(seed_dir.iterdir()):
            if d.name == "oracle" or d.name.startswith("fixed-latency"):
                rows, reason = check_arm_dir(d, EXPECTED_N[part])
                if rows is not None:
                    controls[(cell, d.name)] = build_arm(part, cell, d.name, rows)
    return arms, gaps, controls


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def cell_row(arm: dict[str, Any], anchor: dict[str, Any] | None, cache: dict[int, np.ndarray]) -> dict[str, Any]:
    part, rows, sup = arm["part"], arm["rows"], arm["supported"]
    counts = draws_for(arm, cache)
    blk = arm["block"]
    out: dict[str, Any] = dict(
        part=part, cell=arm["cell"], model=arm["model"], deployment=deployment(arm["model"]),
        n_arrivals=len(rows), n_supported=int(sup.sum()),
        primary_metric="strict" if part == "A" else "operational",
    )
    for metric in ("strict", "operational"):
        pt = rate(arm[metric], sup)
        lo, hi = ci(boot_rate(arm[metric], sup, blk, counts), pt)
        out.update({f"{metric}_completion": pt, f"{metric}_ci_low": lo, f"{metric}_ci_high": hi})
    n = len(rows)
    for s in STATUS_ALL:
        out[f"share_{s}"] = sum(x == s for x in arm["status"]) / n
    shared = None
    if anchor is not None:
        if anchor["ids"] != arm["ids"]:
            raise ValueError(f"{arm['cell']}: {arm['model']} and anchor are not on the same trace")
        shared = arm["primary"] & anchor["primary"]
    for label, mask in (("completions", arm["primary"]), ("shared", shared)):
        vals = arm["T"][mask] if mask is not None else np.array([])
        for q in (50, 95, 99):
            pt = percentile(vals, q) if mask is not None else None
            out[f"T_p{q}_{label}"] = pt
            if q in (50, 95):
                lo, hi = ci(boot_percentile(vals, blk[mask], counts, q), pt) if mask is not None else (None, None)
                out[f"T_p{q}_{label}_ci_low"], out[f"T_p{q}_{label}_ci_high"] = lo, hi
    out["n_shared_with_anchor"] = int(shared.sum()) if shared is not None else None
    out["shared_anchor"] = anchor["model"] if anchor is not None else None
    costs = [r.get("cost_usd") for r in rows]
    n_correct = int(arm["primary"].sum())
    known = float(sum(c for c in costs if c is not None))
    out["cost_usd_known"] = known
    out["unknown_cost_calls"] = sum(1 for r in rows if r.get("api_calls") and r.get("cost_usd") is None)
    out["usd_per_correct_completion"] = known / n_correct if deployment(arm["model"]) == "hosted" and n_correct else None
    out["api_calls"] = int(sum(int(r.get("api_calls") or 0) for r in rows))
    out["cache_hits"] = int(sum(bool(r.get("cache_hit")) for r in rows))
    out["locality_violations"] = int(sum(bool(r.get("locality_violation")) for r in rows))
    if part == "B":
        out["offsite_bytes"] = int(sum(int(r["score"].get("offsite_violation_bytes") or 0) for r in rows))
        out["false_admissions"] = int(sum(false_admission_b(r) for r in rows))
        out["correct_rejections"] = int(sum(bool(r["score"].get("correct_rejection")) for r in rows))
    else:
        out["offsite_bytes"] = out["false_admissions"] = out["correct_rejections"] = None
    decided = [r for r in rows if r.get("decision_start") is not None]
    out["mean_admission_wait_s"] = _mean([r.get("queue_wait_s") for r in decided])
    out["mean_decision_s"] = _mean([r.get("decision_elapsed_s") for r in decided])
    if part == "A":
        served = [r for r in rows if r["status"] in SERVED_A]
        out["mean_service_queue_s"] = _mean([r["service_queue_s"] for r in served])
        out["mean_transfer_s"] = _mean([r["transfer_s"] for r in served])
        out["mean_execution_s"] = _mean([r["service_s"] for r in served])
    else:
        served = [r for r in rows if r.get("service_dispatch") is not None]
        out["mean_service_queue_s"] = _mean([r["service_dispatch"] - r["service_enqueued"] for r in served])
        out["mean_transfer_s"] = None  # testbed round trip includes the link; not separable
        out["mean_execution_s"] = _mean([r.get("origin_service_s") for r in served])  # includes link
    return out


def gap_row(part: str, cell: str, model: str, anchor: str, st: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = dict(part=part, cell=cell, model=model, anchor=anchor, deployment=deployment(model))
    for key in ("G_primary", "G_operational", "dT95", "dT50"):
        lo, hi = ci(st[f"{key}_boot"], st[key])
        out.update({key: st[key], f"{key}_ci_low": lo, f"{key}_ci_high": hi})
        if key in ("G_primary", "dT95"):
            out[f"{key}_p"] = pvalue(st[f"{key}_boot"]) if st[key] is not None else None
    out["n_shared"] = st["n_shared"]
    out["h7_eligible"] = h7_eligible(st["n_shared"])
    return out


# ---------------------------------------------------------------------------
# Hypotheses
# ---------------------------------------------------------------------------

def h7_eligible(n_shared: int) -> bool:
    return n_shared >= MIN_SHARED


def _test(hyp: str, part: str, model: str, anchor: str, cell: str, statistic: str,
          point: float | None, boot: np.ndarray | None, n_shared: int | None = None,
          eligible: bool = True, note: str = "") -> dict[str, Any]:
    lo, hi = ci(boot, point) if boot is not None else (None, None)
    return dict(
        row_type="test", hypothesis=hyp, part=part, contrast=model, anchor=anchor, cell=cell, statistic=statistic,
        point=point, ci_low=lo, ci_high=hi,
        p_raw=pvalue(boot) if boot is not None and point is not None else None,
        p_holm=None, holm_family_size=None, n_shared=n_shared, eligible=eligible, resolved=None, note=note,
    )


def apply_holm(tests: list[dict[str, Any]]) -> None:
    """Holm within each (hypothesis, part) over the evaluable, eligible tests; then set `resolved`."""
    fams: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for t in tests:
        if t["eligible"] and t["p_raw"] is not None:
            fams.setdefault((t["hypothesis"], t["part"]), []).append(t)
    for fam in fams.values():
        for t, p in zip(fam, holm_adjust([t["p_raw"] for t in fam])):
            t["p_holm"], t["holm_family_size"] = p, len(fam)
    for t in tests:
        if t["p_holm"] is None:
            t["resolved"] = None
        else:
            t["resolved"] = bool((t["ci_low"] > 0 or t["ci_high"] < 0) and t["p_holm"] < ALPHA)


def h6_tests(arms: dict[tuple[str, str], dict[str, Any]], cache: dict[int, np.ndarray]) -> list[dict[str, Any]]:
    x = np.log2([lam for _, lam in LOAD_CELLS])
    w = (x - x.mean()) / ((x - x.mean()) ** 2).sum()
    tests = []
    for model, anchor in [(m, HOSTED_ANCHOR) for m in HOSTED_CONTRASTS] + [(SELF_CONTRAST, SELF_ANCHOR)]:
        missing = [c for c, _ in LOAD_CELLS if (c, model) not in arms or (c, anchor) not in arms]
        if missing:
            tests.append(_test("H6", "A", model, anchor, "load_sweep", "slope_G_on_log2_lambda", None, None,
                               note="missing arms: " + ";".join(missing)))
            continue
        stats = [pair_stats(arms[(c, model)], arms[(c, anchor)], cache) for c, _ in LOAD_CELLS]
        point = float(sum(wk * s["G_primary"] for wk, s in zip(w, stats)))
        boot = sum(wk * s["G_primary_boot"] for wk, s in zip(w, stats))
        tests.append(_test("H6", "A", model, anchor, "load_sweep", "slope_G_on_log2_lambda", point, boot))
    return tests


def h7_tests(arms_by_part: dict[str, dict], cache_by_part: dict[str, dict]) -> list[dict[str, Any]]:
    tests = []
    for part in ("A", "B"):
        arms = arms_by_part[part]
        for cell in CACHE_OFF[part]:
            for model in HOSTED_CONTRASTS:
                if (cell, model) not in arms or (cell, HOSTED_ANCHOR) not in arms:
                    tests.append(_test("H7", part, model, HOSTED_ANCHOR, cell, "dT_p95_shared", None, None,
                                       eligible=False, note="missing arm"))
                    continue
                st = pair_stats(arms[(cell, model)], arms[(cell, HOSTED_ANCHOR)], cache_by_part[part])
                elig = h7_eligible(st["n_shared"])
                tests.append(_test("H7", part, model, HOSTED_ANCHOR, cell, "dT_p95_shared", st["dT95"],
                                   st["dT95_boot"], n_shared=st["n_shared"], eligible=elig,
                                   note="" if elig else f"ineligible: shared successes {st['n_shared']} < {MIN_SHARED}"))
    return tests


def h8_tests(arms_by_part: dict[str, dict], cache_by_part: dict[str, dict]) -> list[dict[str, Any]]:
    tests = []
    for part in ("A", "B"):
        arms = arms_by_part[part]
        for off, on in H8_PAIRS[part]:
            for model in HOSTED_CONTRASTS:
                cell = f"{on}-minus-{off}"
                need = [(c, m) for c in (off, on) for m in (model, HOSTED_ANCHOR)]
                if any(k not in arms for k in need):
                    for stat in ("did_abs_G", "did_abs_dT_p50_shared"):
                        tests.append(_test("H8", part, model, HOSTED_ANCHOR, cell, stat, None, None,
                                           note="missing arm"))
                    continue
                s_off = pair_stats(arms[(off, model)], arms[(off, HOSTED_ANCHOR)], cache_by_part[part])
                s_on = pair_stats(arms[(on, model)], arms[(on, HOSTED_ANCHOR)], cache_by_part[part])
                for stat, key in (("did_abs_G", "G_primary"), ("did_abs_dT_p50_shared", "dT50")):
                    if s_on[key] is None or s_off[key] is None:
                        tests.append(_test("H8", part, model, HOSTED_ANCHOR, cell, stat, None, None,
                                           note="no shared successes"))
                        continue
                    point = abs(s_on[key]) - abs(s_off[key])
                    boot = np.abs(s_on[f"{key}_boot"]) - np.abs(s_off[f"{key}_boot"])
                    tests.append(_test("H8", part, model, HOSTED_ANCHOR, cell, stat, point, boot,
                                       n_shared=min(s_on["n_shared"], s_off["n_shared"])))
    return tests


def _verdict(hyp: str, part: str, model: str, anchor: str, verdict: str, complete: bool, **kw: Any) -> dict[str, Any]:
    return dict(row_type="verdict", hypothesis=hyp, part=part, contrast=model, anchor=anchor, verdict=verdict,
                coverage_complete=complete, **kw)


def h6_verdicts(tests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for t in tests:
        if t["hypothesis"] != "H6":
            continue
        if t["resolved"] is None:
            out.append(_verdict("H6", "A", t["contrast"], t["anchor"], "not evaluable", False, n_tests=0))
            continue
        holds = t["resolved"] and t["point"] > 0
        out.append(_verdict("H6", "A", t["contrast"], t["anchor"], "holds" if holds else "does not hold", True,
                            n_tests=1, n_expected_resolved=int(holds)))
    return out


def h7_verdicts(tests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for model in HOSTED_CONTRASTS:
        mine = [t for t in tests if t["hypothesis"] == "H7" and t["contrast"] == model]
        missing = [f"{t['part']}:{t['cell']}" for t in mine if t["n_shared"] is None]
        elig = [t for t in mine if t["n_shared"] is not None and h7_eligible(t["n_shared"]) and t["point"] is not None]
        inelig = [f"{t['part']}:{t['cell']}" for t in mine if t["n_shared"] is not None and not h7_eligible(t["n_shared"])]
        pos = sum(bool(t["resolved"]) and t["point"] > 0 for t in elig)
        rev = sum(bool(t["resolved"]) and t["point"] < 0 for t in elig)
        holds = len(elig) >= MIN_ELIGIBLE and pos >= H7_SHARE * len(elig) and rev == 0
        verdict = "holds" if holds else ("not evaluable" if len(elig) < MIN_ELIGIBLE else "does not hold")
        out.append(_verdict("H7", "A+B", model, HOSTED_ANCHOR, verdict, not missing, n_tests=len(elig),
                            n_eligible=len(elig), n_expected_resolved=pos, n_reversed_resolved=rev,
                            ineligible_cells=";".join(inelig), missing_cells=";".join(missing)))
    return out


def h8_verdicts(tests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for part in ("A", "B"):
        for model in HOSTED_CONTRASTS:
            mine = [t for t in tests if t["hypothesis"] == "H8" and t["part"] == part and t["contrast"] == model]
            done = [t for t in mine if t["resolved"] is not None]
            missing = sorted({t["cell"] for t in mine if t["resolved"] is None})
            pos = sum(t["resolved"] and t["point"] < 0 for t in done)
            if not done:
                verdict = "not evaluable"
            else:
                verdict = "holds" if pos == len(done) else "does not hold"
            out.append(_verdict("H8", part, model, HOSTED_ANCHOR, verdict, not missing, n_tests=len(done),
                                n_expected_resolved=pos, missing_cells=";".join(missing)))
    return out


# ---------------------------------------------------------------------------
# Controls and topology replay
# ---------------------------------------------------------------------------

def control_rows(controls: dict[str, dict], arms: dict[str, dict]) -> list[dict[str, Any]]:
    out = []
    for part in ("A", "B"):
        for (cell, model), arm in sorted(controls[part].items()):
            ceiling = [rate(a["primary"], a["supported"]) for (c, _), a in arms[part].items() if c == cell]
            prim = rate(arm["primary"], arm["supported"])
            row = dict(part=part, cell=cell, model=model,
                       delay_s=float(model.rsplit("-", 1)[1]) if model.startswith("fixed-latency-") else None,
                       strict_completion=rate(arm["strict"], arm["supported"]),
                       operational_completion=rate(arm["operational"], arm["supported"]),
                       max_interpreter_primary=max(ceiling) if ceiling else None, flag="")
            if model == "oracle" and ceiling and prim is not None and prim < max(ceiling):
                row["flag"] = "oracle below an interpreter: not the feasibility ceiling"
            out.append(row)
    fixed = sorted((r for r in out if r["part"] == "A" and r["cell"] == "load_16" and r["delay_s"] is not None),
                   key=lambda r: r["delay_s"])
    for a, b in zip(fixed, fixed[1:]):
        if not b["strict_completion"] < a["strict_completion"]:
            b["flag"] = f"completion does not fall from delay {a['delay_s']} to {b['delay_s']}"
    return out


def topology_replay(admission_by_model: dict[str, list[dict[str, Any]]],
                    traces_by_n: dict[int, list[dict[str, Any]]],
                    arms_a: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    """Exploratory: replay each interpreter's load_8 admission timeline through the simulator for N in TOPOLOGY_N."""
    base = traces_by_n[5]
    out = []
    for n in TOPOLOGY_N:
        tr = traces_by_n[n]
        if [(r["id"], r["arrival"], r["text"], r["deadline"]) for r in tr] != \
                [(r["id"], r["arrival"], r["text"], r["deadline"]) for r in base]:
            raise ValueError(f"topology trace N={n} does not share arrivals/texts with load_8")
    for model, adm in admission_by_model.items():
        ctrace = {r["id"]: r for r in adm}
        for n in TOPOLOGY_N:
            sim_rows = simulate(traces_by_n[n], controller_trace=ctrace, deadline=2.0, n_edges=n)
            for r in sim_rows:
                r["truth"] = _truth(r)
            arm = build_arm("A", f"replay_{n}", model, sim_rows)
            live_cell = "load_8" if n == 5 else f"topo_{n}"
            live = arms_a.get((live_cell, model))
            row = dict(
                analysis="exploratory", model=model, deployment=deployment(model), N=n,
                admission_source="load_8", origins_source=live_cell, n_supported=int(arm["supported"].sum()),
                strict_completion=rate(arm["strict"], arm["supported"]),
                operational_completion=rate(arm["operational"], arm["supported"]),
                live_strict_completion=rate(live["strict"], live["supported"]) if live else None,
                live_operational_completion=rate(live["operational"], live["supported"]) if live else None,
                replay_reproduces_live=None,
            )
            if n == 5 and live is not None:
                row["replay_reproduces_live"] = arm["status"] == live["status"]
            out.append(row)
    return out


# ---------------------------------------------------------------------------
# CSV / Markdown
# ---------------------------------------------------------------------------

def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "true" if v else "false"
    if isinstance(v, (float, np.floating)):
        return "" if not math.isfinite(v) else f"{float(v):.6g}"
    return str(v)


def to_csv(rows: list[dict[str, Any]], columns: list[str] | None = None) -> str:
    if columns is None:
        columns = []
        for r in rows:
            columns.extend(k for k in r if k not in columns)
    buf = StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(columns)
    for r in rows:
        w.writerow([_fmt(r.get(c)) for c in columns])
    return buf.getvalue()


def read_csv(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


HYP_COLUMNS = [
    "row_type", "hypothesis", "part", "contrast", "anchor", "cell", "statistic", "point", "ci_low", "ci_high",
    "p_raw", "p_holm", "holm_family_size", "n_shared", "eligible", "resolved", "verdict", "coverage_complete",
    "n_tests", "n_eligible", "n_expected_resolved", "n_reversed_resolved", "ineligible_cells", "missing_cells", "note",
]


def _num(s: str, nd: int = 3) -> str:
    if s == "":
        return "–"
    x = float(s)
    return f"{x:.2e}" if x != 0 and abs(x) < 10 ** -nd else f"{x:.{nd}f}"


def _p(s: str) -> str:
    return "–" if s == "" else f"{float(s):.3g}"


def render_hypotheses_md(hyp_csv: str, gaps: list[dict[str, str]], notes: list[str]) -> str:
    """Render hypotheses.md from the text of hypotheses.csv; every result sentence is computed from its rows."""
    rows = read_csv(hyp_csv)
    verdicts = [r for r in rows if r["row_type"] == "verdict"]
    tests = [r for r in rows if r["row_type"] == "test"]
    lines = ["# RQ5 hypotheses H6–H8 (EXP-2026-002)", "",
             "Generated by `scripts/eb_rq5_analyze.py` from `hypotheses.csv`. Resolved = 95% block-bootstrap CI "
             "excludes 0 and Holm-adjusted p < 0.05 (Holm within each hypothesis and part).", ""]
    if gaps or notes:
        lines += ["## Coverage gaps (`--allow-partial`): results below are PROVISIONAL", ""]
        lines += [f"- Part {g['part']} / {g['cell']} / {g['model']}: {g['reason']}" for g in gaps]
        lines += [f"- {n}" for n in notes]
        lines.append("")
    lines += ["## Verdicts", "", "| Hypothesis | Part | Contrast | Verdict | Basis | Coverage complete |",
              "|---|---|---|---|---|---|"]
    for v in verdicts:
        if v["hypothesis"] == "H7":
            basis = (f"{v['n_expected_resolved']}/{v['n_eligible']} eligible cells resolved ΔT>0; "
                     f"{v['n_reversed_resolved']} reversed; ineligible: {v['ineligible_cells'] or 'none'}")
        else:
            basis = f"{v['n_expected_resolved'] or 0}/{v['n_tests']} tests resolved in the registered direction"
        if v["missing_cells"]:
            basis += f"; missing: {v['missing_cells']}"
        lines.append(f"| {v['hypothesis']} | {v['part']} | {v['contrast']} vs {v['anchor']} | {v['verdict']} | "
                     f"{basis} | {v['coverage_complete']} |")
    lines += ["", "## Summary", ""]
    for hyp in ("H6", "H7", "H8"):
        vs = [v for v in verdicts if v["hypothesis"] == hyp]
        held = [f"{v['contrast']} (Part {v['part']})" for v in vs if v["verdict"] == "holds"]
        lines.append(f"- {hyp}: holds for {len(held)} of {len(vs)} verdicts"
                     + (f" ({'; '.join(held)})" if held else "") + ".")
    lines += ["", "## Tests", "",
              "| Hyp | Part | Contrast | Cell | Statistic | Point | 95% CI | p (raw) | p (Holm, m) | Shared N | Eligible | Resolved | Note |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in tests:
        ci_txt = f"[{_num(t['ci_low'])}, {_num(t['ci_high'])}]" if t["ci_low"] else "–"
        holm = f"{_p(t['p_holm'])} ({t['holm_family_size']})" if t["p_holm"] else "–"
        lines.append(f"| {t['hypothesis']} | {t['part']} | {t['contrast']} | {t['cell']} | {t['statistic']} | "
                     f"{_num(t['point'])} | {ci_txt} | {_p(t['p_raw'])} | {holm} | {t['n_shared'] or '–'} | "
                     f"{t['eligible']} | {t['resolved'] or '–'} | {t['note']} |")
    return "\n".join(lines) + "\n"


def paper_numbers_md(tables: dict[str, str]) -> str:
    """Every quotable RQ5 number with the CSV file, data row (1-based) and column it comes from."""
    spec = {
        "cells.csv": (["part", "cell", "model"], ["strict_completion", "strict_ci_low", "strict_ci_high",
                      "operational_completion", "operational_ci_low", "operational_ci_high", "T_p50_shared",
                      "T_p95_shared", "T_p99_shared", "T_p95_completions", "usd_per_correct_completion",
                      "locality_violations", "false_admissions", "correct_rejections", "share_invalid_decision"]),
        "gaps.csv": (["part", "cell", "model", "anchor"], ["G_primary", "G_primary_ci_low", "G_primary_ci_high",
                     "dT95", "dT95_ci_low", "dT95_ci_high", "n_shared"]),
        "hypotheses.csv": (["row_type", "hypothesis", "part", "contrast", "cell", "statistic"],
                           ["point", "ci_low", "ci_high", "p_holm", "verdict"]),
        "topology_replay.csv": (["model", "N"], ["strict_completion", "operational_completion"]),
        "controls.csv": (["part", "cell", "model"], ["strict_completion", "operational_completion", "flag"]),
    }
    lines = ["# RQ5 paper numbers (EXP-2026-002)", "",
             "Each number: `file` row (1-based data row, header excluded) and column. Quote only from here.", ""]
    for name, (keys, cols) in spec.items():
        rows = read_csv(tables[name])
        lines += [f"## {name}", ""]
        if not rows:
            lines += ["(no rows)", ""]
            continue
        for i, r in enumerate(rows, 1):
            key = ", ".join(f"{k}={r[k]}" for k in keys if r.get(k))
            vals = "; ".join(f"{c}={r[c]}" for c in cols if r.get(c, "") != "")
            if vals:
                lines.append(f"- `{name}` row {i} ({key}): {vals}")
        lines.append("")
    return "\n".join(lines)


def run_analysis(root_a: Path, root_b: Path, traces_dir: Path) -> dict[str, Any]:
    arms_a, gaps_a, ctl_a = load_part(root_a, "A")
    arms_b, gaps_b, ctl_b = load_part(root_b, "B")
    arms = {"A": arms_a, "B": arms_b}
    cache: dict[str, dict[int, np.ndarray]] = {"A": {}, "B": {}}

    cells, gaps_rows = [], []
    for part in ("A", "B"):
        for cell in CELLS[part]:
            for model in MODELS:
                arm = arms[part].get((cell, model))
                if arm is None:
                    continue
                anc = arms[part].get((cell, anchor_for(model)))
                cells.append(cell_row(arm, anc, cache[part]))
                if anc is not None and model != anchor_for(model):
                    st = pair_stats(arm, anc, cache[part])
                    gaps_rows.append(gap_row(part, cell, model, anchor_for(model), st))

    tests = h6_tests(arms_a, cache["A"]) + h7_tests(arms, cache) + h8_tests(arms, cache)
    apply_holm(tests)
    verdicts = h6_verdicts(tests) + h7_verdicts(tests) + h8_verdicts(tests)

    traces_by_n = {n: _read_jsonl(traces_dir / ("load_8" if n == 5 else f"topo_{n}") / "1.jsonl") for n in TOPOLOGY_N}
    adm = {}
    for model in MODELS:
        if ("load_8", model) in arms_a:
            adm[model] = _read_jsonl(root_a / "load_8" / "seed_1" / model / "admission.jsonl")
    replay = topology_replay(adm, traces_by_n, arms_a)
    controls = control_rows({"A": ctl_a, "B": ctl_b}, arms)
    notes = []
    if not controls:
        notes.append("positive controls (oracle, fixed-latency-*): no complete control arm under the run roots")
    return dict(cells=cells, gaps=gaps_rows, hypotheses=tests + verdicts, replay=replay, controls=controls,
                coverage_gaps=gaps_a + gaps_b, notes=notes)
