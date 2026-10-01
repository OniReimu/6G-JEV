"""EXP-2026-003 C1 analysis pipeline: RQ4 / H3, near-RT feasibility, C1 latency, quality, and cost.

Pre-registered analysis plan (v0.2 frozen, seed 20260925, 10,000 resamples).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass, field
import io
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

# Sister analysis code helpers (reused without modification)
from src.edgebench.scoring import (
    bootstrap_ci,
    exact_mcnemar_test,
    cochrans_q,
    holm_adjust,
    invert_bootstrap_pvalue,
)
from src.edgebench.analysis import (
    parse_header_tz,
    parse_power_trace,
    integrate_energy,
)

# ---------------------------------------------------------------------------
# Constants & Model Definitions
# ---------------------------------------------------------------------------

DELTA_E2_S: float = 0.005
NEAR_RT_DEADLINE_S: float = 1.0
DEFAULT_SEED: int = 20260925
DEFAULT_RESAMPLES: int = 10000
ERROR_RATE_THRESHOLD: float = 0.05

THE_SIX_MODELS: list[str] = [
    "Jev-1.13.0",
    "DeepSeek-V4.1-Flash",
    "GLM-5.3-Flash",
    "Qwen3.8-Flash",
    "SemIf-Qwen3.5-4B",
    "AnyJev-L0",
]

REFERENCE_MODELS: list[str] = [
    "Qwen3.5-4B-JSON",
]

EXPECTED_MODELS: list[str] = THE_SIX_MODELS + REFERENCE_MODELS

HOSTED_MODELS: list[str] = [
    "Jev-1.13.0",
    "DeepSeek-V4.1-Flash",
    "GLM-5.3-Flash",
    "Qwen3.8-Flash",
]

SELFHOSTED_MODELS: list[str] = [
    "SemIf-Qwen3.5-4B",
    "AnyJev-L0",
    "Qwen3.5-4B-JSON",
]

MODEL_ALIASES: dict[str, str] = {
    "AnyJev": "AnyJev-L0",
    "AnyJev (L0)": "AnyJev-L0",
    "anyjev-l0": "AnyJev-L0",
    "anyjev": "AnyJev-L0",
}

FIELDS: tuple[str, ...] = (
    "action",
    "class",
    "scope",
    "target_cluster",
    "priority",
    "prb_share",
    "edge_site",
    "latency_target",
    "duration",
)

ACTUATED_FIELDS: tuple[str, ...] = (
    "action",
    "class",
    "scope",
    "priority",
)

CELL_COUNTS: list[int] = [3, 7, 21, 57]
QUALITIES: list[str] = ["fresh", "stale", "noisy", "contradictory"]

EXPECTED_CONDITIONS: list[str] = [
    f"c{c}_{q}" for c in CELL_COUNTS for q in QUALITIES
]

# Original main-run directories (execution log): the block under the run directory with this
# name is the first primary candidate for every cell of a model with that deployment.
MAIN_RUN_DIRS: dict[str, str] = {
    "hosted": "c1_hosted",
    "selfhosted": "c1_selfhosted",
}

# D-3 (deviations.md): Qwen3.8-Flash's primary row is chosen per case across its blocks, not per block.
PER_CASE_PRIMARY_MODELS: frozenset[str] = frozenset({"Qwen3.8-Flash"})
PER_CASE_PRIMARY_LABEL: str = "per-case (D-3)"

# RANIntent v1 design: 300 test cases per condition, 150 of them state-dependent.
CASES_PER_CONDITION: int = 300
STATE_DEPENDENT_PER_CONDITION: int = 150


def canonical_model_name(name: str) -> str:
    """Strip platform/version suffixes and map aliases to canonical names."""
    base = name.split("@")[0].strip()
    return MODEL_ALIASES.get(base, base)


def model_deployment(model: str, rows: list[dict[str, Any]] | None = None) -> str | None:
    """'hosted' or 'selfhosted' from the roster; unknown models fall back to the rows' platform."""
    if model in HOSTED_MODELS:
        return "hosted"
    if model in SELFHOSTED_MODELS:
        return "selfhosted"
    for r in rows or []:
        if r.get("platform"):
            return "hosted" if r["platform"] == "hosted" else "selfhosted"
    return None


def _read_jsonl_strict(path: Path) -> list[dict[str, Any]]:
    """Parse a JSONL file; any malformed line raises with file and line number.

    A malformed final line without a trailing newline is reported as a torn last line
    (typical of an interrupted run); it is never dropped silently.
    """
    text = Path(path).read_text(encoding="utf-8")
    lines = text.split("\n")
    rows: list[dict[str, Any]] = []
    for i, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            torn = i == len(lines) and not text.endswith("\n")
            kind = "torn last line (interrupted run?)" if torn else "malformed JSON"
            raise ValueError(f"{path}:{i}: {kind}: {exc}") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"{path}:{i}: expected a JSON object, got {type(obj).__name__}")
        rows.append(obj)
    return rows


# ---------------------------------------------------------------------------
# Corpus & Case Loading
# ---------------------------------------------------------------------------

@dataclass
class CorpusCase:
    case_id: str
    condition: str
    n_cells: int
    quality: str
    split: str
    half: str
    truth: dict[str, str]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def is_state_dependent(self) -> bool:
        return self.half == "state_dependent"


def load_corpus_cases(corpus_dir: Path | str) -> dict[str, dict[str, CorpusCase]]:
    """Load test cases from the corpus directory for all conditions.

    Returns dict mapping condition -> {case_id: CorpusCase}.
    """
    corpus_dir = Path(corpus_dir)
    cases_by_cond: dict[str, dict[str, CorpusCase]] = defaultdict(dict)

    # Search in corpus_dir / "RQ4" / cond / "test.jsonl" and corpus_dir / cond / "test.jsonl"
    for cond in EXPECTED_CONDITIONS:
        candidates = [
            corpus_dir / "RQ4" / cond / "test.jsonl",
            corpus_dir / cond / "test.jsonl",
        ]
        test_file = None
        for c in candidates:
            if c.is_file():
                test_file = c
                break
        if test_file is None:
            # Fallback: search anywhere under corpus_dir for test.jsonl matching condition
            matches = list(corpus_dir.rglob(f"{cond}/**/test.jsonl"))
            if matches:
                test_file = matches[0]

        if test_file is not None and test_file.is_file():
            with open(test_file, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        cid = str(data["case_id"])
                        cases_by_cond[cond][cid] = CorpusCase(
                            case_id=cid,
                            condition=str(data.get("condition", cond)),
                            n_cells=int(data.get("n_cells", 0)),
                            quality=str(data.get("quality", "")),
                            split=str(data.get("split", "test")),
                            half=str(data.get("half", "state_dependent" if "state_dependent" in str(data) else "named_scope")),
                            truth=dict(data.get("truth", {})),
                            meta=dict(data.get("meta", {})),
                        )

    return cases_by_cond


# ---------------------------------------------------------------------------
# File & Ledger Discovery
# ---------------------------------------------------------------------------

def _block_label(path: Path, root: Path) -> str:
    """Block label = run-directory name, plus the sub-path when the file sits below it."""
    rel = path.parent.relative_to(root)
    return root.name if str(rel) == "." else f"{root.name}/{rel.as_posix()}"


def find_run_files(run_dirs: list[Path | str]) -> dict[str, Any]:
    """Recursively discover ledgers, probes, and power traces across run dirs.

    Also returns `block_of`: ledger/probe path -> (run-directory name, block label). Every
    directory holding a ledger is one block; the run-directory name decides whether it is the
    original main run (MAIN_RUN_DIRS).
    """
    ledgers: set[Path] = set()
    probes: set[Path] = set()
    power_traces: set[Path] = set()
    block_of: dict[Path, tuple[str, str]] = {}

    for rd in run_dirs:
        p = Path(rd).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Run directory does not exist: {rd}")
        if p.is_file():
            root = p.parent
            if p.name == "ledger.jsonl":
                ledgers.add(p)
                block_of[p] = (root.name, root.name)
            elif p.name == "probes.jsonl":
                probes.add(p)
                block_of[p] = (root.name, root.name)
            elif "power" in p.name and p.suffix == ".csv":
                power_traces.add(p)
            continue

        for f in p.rglob("ledger.jsonl"):
            ledgers.add(f)
            block_of[f] = (p.name, _block_label(f, p))
        for f in p.rglob("probes.jsonl"):
            probes.add(f)
            block_of[f] = (p.name, _block_label(f, p))
        for f in p.rglob("*power*.csv"):
            power_traces.add(f)

    return {
        "ledgers": sorted(ledgers),
        "probes": sorted(probes),
        "power_traces": sorted(power_traces),
        "block_of": block_of,
    }


def load_probes(
    probe_files: list[Path],
    block_of: dict[Path, tuple[str, str]] | None = None,
) -> dict[tuple[str, str, str], float]:
    """Median successful RTT probe latency per (block, model, condition).

    Only probes with HTTP 200 and no error count. Keyed by block so that a re-run block's
    probes never feed the primary block's net latency.
    """
    block_of = block_of or {}
    lats: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for pf in probe_files:
        pf = Path(pf)
        block = block_of.get(pf, (pf.parent.name, pf.parent.name))[1]
        for r in _read_jsonl_strict(pf):
            if r.get("http_status") != 200 or r.get("error_type") is not None:
                continue
            lat = r.get("latency_s")
            if not isinstance(lat, (int, float)):
                continue
            key = (block, canonical_model_name(str(r.get("model", ""))), str(r.get("condition", "")))
            lats[key].append(float(lat))
    return {k: float(np.median(v)) for k, v in lats.items() if v}


@dataclass
class LoadedRuns:
    primary_by_cell: dict[tuple[str, str], list[dict[str, Any]]]
    sensitivity_by_cell: dict[tuple[str, str], list[dict[str, Any]]]
    blocks_by_cell: dict[tuple[str, str], dict[str, list[dict[str, Any]]]]
    all_rows: list[dict[str, Any]]
    run_sources: dict[str, str]
    primary_block_by_cell: dict[tuple[str, str], str | None] = field(default_factory=dict)
    primary_complete_by_cell: dict[tuple[str, str], bool] = field(default_factory=dict)
    per_case_cells: set[tuple[str, str]] = field(default_factory=set)


def is_transport_ok(r: dict[str, Any]) -> bool:
    """A transport-error-free attempt (HTTP 200, as the runner counts it); schema-invalid answers are outcomes."""
    return r.get("http_status") == 200


def load_all_c1_runs(
    ledger_files: list[Path],
    block_of: dict[Path, tuple[str, str]] | None = None,
    expected_case_ids: dict[str, set[str]] | None = None,
) -> LoadedRuns:
    """Load all ledger rows and pick exactly one primary block per (model, condition).

    Primary-block rule (protocol, Quality controls: "an interpreter arm stopped by a
    transport/protocol error is re-run in a new block; the failed arm is kept for the
    availability table"):
      1. Candidates for a cell are ordered: the original main-run block first (run directory
         MAIN_RUN_DIRS[deployment]: c1_hosted for hosted, c1_selfhosted for self-hosted), then
         every other block by its first send time (t_send_wall), ties by block label.
      2. The primary block is the first candidate that is complete for the cell: it holds a row
         (successful or not) for every corpus case of the condition. A complete arm with
         transport errors stays primary (its errors count in the availability table); an arm
         that stopped before covering every case is not complete, and the next block that
         completes the condition becomes primary.
      3. Every other block of the cell is sensitivity-only.
      4. No candidate complete -> no primary block; the cell is incomplete.
    D-3 per-case rule, for PER_CASE_PRIMARY_MODELS only (Qwen3.8-Flash): with the candidates in the same
    order, a case's primary row is its first transport-error-free attempt (HTTP 200); a case with no such
    attempt keeps its first attempt, a transport failure scored as wrong. The cell is complete when every
    corpus case has a primary row; otherwise it has no primary rows (as in 4; without `expected_case_ids`
    the chosen rows are primary and completeness is >= CASES_PER_CONDITION cases). Every other attempt is
    sensitivity-only and counts in the availability table.
    When `expected_case_ids` has no entry for a condition, completeness cannot be judged: the
    first candidate is primary and completeness is read as >= CASES_PER_CONDITION case ids.

    Within any block the key (model, condition, case_id, repeat) must be unique; a duplicate
    raises ValueError.
    """
    block_of = block_of or {}
    expected_case_ids = expected_case_ids or {}
    by_cell_block: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    root_of_block: dict[str, str] = {}
    all_rows: list[dict[str, Any]] = []
    run_sources: dict[str, str] = {}

    for lf in ledger_files:
        lf = Path(lf)
        root_name, block_name = block_of.get(lf, (lf.parent.name, lf.parent.name))
        if block_name in run_sources:
            raise ValueError(f"Two ledgers map to block '{block_name}': {run_sources[block_name]} and {lf}")
        run_sources[block_name] = str(lf)
        root_of_block[block_name] = root_name

        seen: dict[tuple[str, str, str, Any], int] = {}
        for i, r in enumerate(_read_jsonl_strict(lf), start=1):
            r["_source_file"] = str(lf)
            r["_block_name"] = block_name
            model = canonical_model_name(str(r.get("model", "")))
            r["model"] = model
            cond = str(r.get("condition", ""))
            pk = (model, cond, str(r.get("case_id")), r.get("repeat", 0))
            if pk in seen:
                raise ValueError(
                    f"{lf}: duplicate primary key (model, condition, case_id, repeat)={pk} "
                    f"in block '{block_name}' (rows {seen[pk]} and {i})"
                )
            seen[pk] = i
            all_rows.append(r)
            by_cell_block[(model, cond)][block_name].append(r)

    primary_by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    sensitivity_by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    blocks_by_cell: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    primary_block_by_cell: dict[tuple[str, str], str | None] = {}
    primary_complete_by_cell: dict[tuple[str, str], bool] = {}
    loaded_per_case: set[tuple[str, str]] = set()

    for key, blocks in by_cell_block.items():
        model, cond = key
        blocks_by_cell[key] = blocks
        deployment = model_deployment(model, next(iter(blocks.values())))
        main_dir = MAIN_RUN_DIRS.get(deployment or "", None)

        def order(b: str) -> tuple[int, float, str]:
            sends = [float(r["t_send_wall"]) for r in blocks[b] if isinstance(r.get("t_send_wall"), (int, float))]
            return (0 if root_of_block[b] == main_dir else 1, min(sends) if sends else float("inf"), b)

        candidates = sorted(blocks, key=order)
        expected = expected_case_ids.get(cond)

        if model in PER_CASE_PRIMARY_MODELS:
            loaded_per_case.add(key)
            chosen: dict[str, dict[str, Any]] = {}
            for b in candidates:
                for r in blocks[b]:
                    cid = str(r.get("case_id"))
                    # A ledger row that is not in this condition's corpus can
                    # never become a primary observation.  Keep it only in the
                    # sensitivity/availability evidence below.
                    if expected is not None and cid not in expected:
                        continue
                    if cid not in chosen or (not is_transport_ok(chosen[cid]) and is_transport_ok(r)):
                        chosen[cid] = r
            ids = set(chosen)
            done = len(ids) >= CASES_PER_CONDITION if expected is None else expected <= ids
            # as in the block rule, an unknown case set cannot gate the primary rows (completeness only)
            has_primary = done or expected is None
            primary_ids = {id(r) for r in chosen.values()} if has_primary else set()
            primary_block_by_cell[key] = PER_CASE_PRIMARY_LABEL if has_primary else None
            primary_complete_by_cell[key] = done
            for b in candidates:
                for r in blocks[b]:
                    r["_is_sensitivity"] = id(r) not in primary_ids
                    (sensitivity_by_cell if r["_is_sensitivity"] else primary_by_cell)[key].append(r)
            continue

        def complete(b: str) -> bool:
            ids = {str(r.get("case_id")) for r in blocks[b]}
            if expected is None:
                return len(ids) >= CASES_PER_CONDITION
            return expected <= ids

        if expected is None:
            primary = candidates[0]
        else:
            primary = next((b for b in candidates if complete(b)), None)
        primary_block_by_cell[key] = primary
        primary_complete_by_cell[key] = primary is not None and complete(primary)

        for b in candidates:
            for r in blocks[b]:
                r["_is_sensitivity"] = b != primary
            if b == primary:
                primary_by_cell[key].extend(blocks[b])
            else:
                sensitivity_by_cell[key].extend(blocks[b])

    return LoadedRuns(
        primary_by_cell=primary_by_cell,
        sensitivity_by_cell=sensitivity_by_cell,
        blocks_by_cell=blocks_by_cell,
        all_rows=all_rows,
        run_sources=run_sources,
        primary_block_by_cell=primary_block_by_cell,
        primary_complete_by_cell=primary_complete_by_cell,
        per_case_cells=loaded_per_case,
    )


# ---------------------------------------------------------------------------
# Correctness helpers
# ---------------------------------------------------------------------------

def is_scored_correct(r: dict[str, Any], fields: tuple[str, ...]) -> bool:
    """True when the attempt is schema-valid, has no transport error, and every field is correct.

    Schema-invalid and transport-failed attempts are scored false whatever their `correct` dict says.
    """
    if not bool(r.get("valid")) or r.get("error_type") is not None:
        return False
    corr = r.get("correct") or {}
    return all(bool(corr.get(f, False)) for f in fields)


def is_full_policy_correct(r: dict[str, Any]) -> bool:
    return bool(r.get("valid")) and r.get("error_type") is None and bool(r.get("em", False))


def shared_state_dependent_ids(
    corpus_cases: dict[str, dict[str, CorpusCase]], cond_a: str, cond_b: str
) -> list[str]:
    """The state-dependent tuple ids shared by two conditions; validation fails unless both
    conditions define the same STATE_DEPENDENT_PER_CONDITION ids."""
    for c in (cond_a, cond_b):
        if not corpus_cases.get(c):
            raise ValueError(f"Corpus validation failed: condition '{c}' has no test cases")
    ids_a = {cid for cid, cc in corpus_cases[cond_a].items() if cc.is_state_dependent}
    ids_b = {cid for cid, cc in corpus_cases[cond_b].items() if cc.is_state_dependent}
    if ids_a != ids_b:
        raise ValueError(
            f"Corpus validation failed: state-dependent ids differ between {cond_a} and {cond_b} "
            f"({len(ids_a - ids_b)} only in {cond_a}, {len(ids_b - ids_a)} only in {cond_b})"
        )
    if len(ids_a) != STATE_DEPENDENT_PER_CONDITION:
        raise ValueError(
            f"Corpus validation failed: {cond_a}/{cond_b} define {len(ids_a)} shared state-dependent "
            f"ids, expected {STATE_DEPENDENT_PER_CONDITION}"
        )
    return sorted(ids_a)


# ---------------------------------------------------------------------------
# Metric Aggregations & Analysis
# ---------------------------------------------------------------------------

def compute_availability_table(
    loaded: LoadedRuns,
    expected_models: list[str] = EXPECTED_MODELS,
    expected_conditions: list[str] = EXPECTED_CONDITIONS,
) -> tuple[list[dict[str, Any]], str]:
    """Generate availability table reporting primary and sensitivity blocks per model x condition.

    `total_http_429*` count HTTP 429 over every attempt of the cell (all blocks); `block_attempts` gives
    each block's attempts and HTTP 429 count and rate.
    """
    out_rows: list[dict[str, Any]] = []

    for m in expected_models:
        for c in expected_conditions:
            key = (m, c)
            blocks = loaded.blocks_by_cell.get(key, {})
            p_rows = loaded.primary_by_cell.get(key, [])
            s_rows = loaded.sensitivity_by_cell.get(key, [])
            all_cell_rows = p_rows + s_rows

            p_n = len(p_rows)
            p_errors = sum(1 for r in p_rows if r.get("error_type") is not None)
            p_err_types = Counter(r["error_type"] for r in p_rows if r.get("error_type") is not None)
            p_err_str = "; ".join(f"{k}: {v}" for k, v in sorted(p_err_types.items())) if p_err_types else "None"
            p_rate = (p_errors / p_n) if p_n > 0 else 0.0
            p_avail = (1.0 - p_rate) if p_n > 0 else float("nan")
            flag_gt_5 = p_rate > ERROR_RATE_THRESHOLD

            s_n = len(s_rows)
            s_errors = sum(1 for r in s_rows if r.get("error_type") is not None)
            s_err_types = Counter(r["error_type"] for r in s_rows if r.get("error_type") is not None)
            s_err_str = "; ".join(f"{k}: {v}" for k, v in sorted(s_err_types.items())) if s_err_types else "None"

            tot_n = len(all_cell_rows)
            tot_errors = p_errors + s_errors
            tot_avail = (1.0 - (tot_errors / tot_n)) if tot_n > 0 else float("nan")

            primary_block = loaded.primary_block_by_cell.get(key)
            sens_block_names = sorted(b for b in blocks if b != primary_block)
            # Every block's own error rate against the 5% threshold, so a stopped (non-primary)
            # arm stays visible in the availability table
            flagged_blocks = []
            block_attempts = []
            for b in sorted(blocks):
                b_rows = blocks[b]
                b_rate = sum(1 for r in b_rows if r.get("error_type") is not None) / len(b_rows)
                if b_rate > ERROR_RATE_THRESHOLD:
                    flagged_blocks.append(f"{b} ({b_rate:.1%})")
                b_429 = sum(1 for r in b_rows if r.get("http_status") == 429)
                block_attempts.append(f"{b}: {len(b_rows)} attempts, {b_429} HTTP 429 ({b_429 / len(b_rows):.1%})")
            tot_429 = sum(1 for r in all_cell_rows if r.get("http_status") == 429)
            if not blocks:
                status = "missing"
            elif primary_block is None:
                status = "incomplete"
            else:
                status = "complete" if loaded.primary_complete_by_cell.get(key) else "partial"

            out_rows.append({
                "model": m,
                "condition": c,
                "status": status,
                "primary_rule": "per_case" if key in loaded.per_case_cells else "block",
                "primary_block": primary_block or "None",
                "primary_n": p_n,
                "primary_errors": p_errors,
                "primary_error_types": p_err_str,
                "primary_error_rate": p_rate,
                "primary_avail_rate": p_avail,
                "error_gt_5pct": flag_gt_5,
                "blocks_error_gt_5pct": "; ".join(flagged_blocks) if flagged_blocks else "None",
                "sensitivity_blocks": len(sens_block_names),
                "sensitivity_block_names": "; ".join(sens_block_names) if sens_block_names else "None",
                "sensitivity_n": s_n,
                "sensitivity_errors": s_errors,
                "sensitivity_error_types": s_err_str,
                "total_n": tot_n,
                "total_errors": tot_errors,
                "total_avail_rate": tot_avail,
                "total_http_429": tot_429,
                "total_http_429_rate": (tot_429 / tot_n) if tot_n > 0 else float("nan"),
                "block_attempts": "; ".join(block_attempts) if block_attempts else "None",
            })

    fieldnames = list(out_rows[0].keys()) if out_rows else []
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(out_rows)
    return out_rows, buf.getvalue()


def compute_quality_and_cost_table(
    loaded: LoadedRuns,
    corpus_cases: dict[str, dict[str, CorpusCase]],
    probe_medians: dict[tuple[str, str, str], float] | None = None,
    power_traces: list[Path] | None = None,
    expected_models: list[str] = EXPECTED_MODELS,
    expected_conditions: list[str] = EXPECTED_CONDITIONS,
) -> tuple[list[dict[str, Any]], str]:
    """Per interpreter x condition quality, latency and cost over the primary block.

    - Correctness (field accuracy, actuated-vector, full-policy, target_cluster) is over ALL
      primary attempts (n_attempts); schema-invalid and transport-failed attempts score false.
    - Schema-valid rate = n_valid / n_attempts (valid and no transport error), reported separately.
    - Latency and token statistics are over every primary row that reports them.
    - Net latency (hosted) subtracts the median successful RTT probe of the same block, model and
      condition; without such a probe it is NaN ("no_probe"). Self-hosted: "inapplicable". A D-3
      per-case cell subtracts, row by row, the probe median of the block the row came from
      (rtt_probe_median = median of the subtracted values); any row without one -> "no_probe".
    - Fees per 1,000 correct policies: numerator and denominator over the same cost-reporting rows.
      Self-hosted interpreters carry no API fee: every row counts at 0 USD.
    - Energy per decision only for self-hosted interpreters, integrated from the power trace of
      their own job: the one trace whose directory contains every primary row's ledger. Hosted
      interpreters, and cells with no such trace or more than one, are empty (NA).
    """
    probe_medians = probe_medians or {}
    power_traces = power_traces or []
    out_rows: list[dict[str, Any]] = []
    nan = float("nan")

    # Pre-parse power traces
    parsed_traces: list[dict[str, Any]] = []
    for pt in power_traces:
        try:
            parsed_traces.append(parse_power_trace(pt))
        except Exception:
            pass

    for m in expected_models:
        for c in expected_conditions:
            key = (m, c)
            p_rows = loaded.primary_by_cell.get(key, [])
            n_attempts = len(p_rows)
            n_valid = sum(1 for r in p_rows if bool(r.get("valid")) and r.get("error_type") is None)

            def acc(fields: tuple[str, ...]) -> float:
                return (sum(is_scored_correct(r, fields) for r in p_rows) / n_attempts) if n_attempts else nan

            field_accs = {f: acc((f,)) for f in FIELDS}
            actuated_vector_match = acc(ACTUATED_FIELDS)
            n_full_correct = sum(1 for r in p_rows if is_full_policy_correct(r))
            full_policy_match = (n_full_correct / n_attempts) if n_attempts else nan
            schema_valid_rate = (n_valid / n_attempts) if n_attempts else nan
            unsafe_rate = (
                sum(1 for r in p_rows if bool(r.get("valid")) and r.get("error_type") is None and bool(r.get("unsafe_locality"))) / n_valid
            ) if n_valid else nan

            # target_cluster on the corpus state-dependent cases of this condition; a case without
            # a primary attempt counts as wrong. No corpus for the condition -> NaN.
            state_dep_cids = sorted(cid for cid, cc in corpus_cases.get(c, {}).items() if cc.is_state_dependent)
            p_rows_map = {str(r["case_id"]): r for r in p_rows}
            if state_dep_cids and n_attempts:
                target_cluster_state_dep_acc = sum(
                    is_scored_correct(p_rows_map[cid], ("target_cluster",))
                    for cid in state_dep_cids if cid in p_rows_map
                ) / len(state_dep_cids)
            else:
                target_cluster_state_dep_acc = nan

            lats = [float(r["latency_s"]) for r in p_rows if r.get("latency_s") is not None]
            p50, p95, p99 = np.percentile(lats, [50, 95, 99]) if lats else (nan, nan, nan)

            deployment = model_deployment(m, p_rows)
            if deployment == "selfhosted":
                rtt_median = nan
                net_status = "inapplicable_selfhosted"
            elif key in loaded.per_case_cells:
                rtts = [probe_medians.get((r.get("_block_name", ""), m, c), nan)
                        for r in p_rows if r.get("latency_s") is not None]
                ok = bool(rtts) and not any(np.isnan(x) for x in rtts)
                rtt_median = float(np.median(rtts)) if ok else nan
                net_status = "ok" if ok else ("no_probe" if n_attempts else "no_data")
            else:
                rtt_median = probe_medians.get((loaded.primary_block_by_cell.get(key) or "", m, c), nan)
                rtts = [rtt_median] * len(lats)
                net_status = "ok" if not np.isnan(rtt_median) else ("no_probe" if n_attempts else "no_data")
            if net_status == "ok" and lats:
                net_p50, net_p95, net_p99 = np.percentile(
                    [max(0.0, lat - rtt) for lat, rtt in zip(lats, rtts)], [50, 95, 99])
            else:
                net_p50 = net_p95 = net_p99 = nan

            in_tokens = [int(r["input_tokens"]) for r in p_rows if r.get("input_tokens") is not None]
            out_tokens = [int(r["output_tokens"]) for r in p_rows if r.get("output_tokens") is not None]
            mean_in = float(np.mean(in_tokens)) if in_tokens else nan
            mean_out = float(np.mean(out_tokens)) if out_tokens else nan

            # Fees per 1,000 full-policy-correct policies, over the rows that report a cost
            if deployment == "selfhosted":
                cost_rows = list(p_rows)
                total_cost = 0.0
            else:
                cost_rows = [r for r in p_rows if r.get("cost_usd") is not None]
                total_cost = sum(float(r["cost_usd"]) for r in cost_rows)
            # Plan (analysis-plan.md Definitions): a policy is correct when the actuated vector matches.
            n_correct_costed = sum(1 for r in cost_rows if is_scored_correct(r, ACTUATED_FIELDS))
            cost_per_1k = (total_cost / n_correct_costed * 1000.0) if n_correct_costed else nan

            # Energy per decision (self-hosted)
            energy_per_dec_str = ""
            own_traces = [
                tr for tr in parsed_traces
                if all(Path(tr["file"]).resolve().parent in Path(r["_source_file"]).resolve().parents
                       for r in p_rows)
            ] if deployment == "selfhosted" and p_rows else []
            if len(own_traces) == 1:
                t_sends = [r["t_send_wall"] for r in p_rows if r.get("t_send_wall")]
                t_recvs = [r["t_recv_wall"] for r in p_rows if r.get("t_recv_wall")]
                if t_sends and t_recvs:
                    t_start = min(t_sends)
                    t_end = max(t_recvs)
                    for tr in own_traces:
                        if len(tr["times"]) and tr["times"][0] <= t_start <= tr["times"][-1]:
                            e_res = integrate_energy(
                                tr["times"],
                                tr["powers"],
                                t_start=t_start,
                                t_end=t_end,
                                n_decisions=n_valid,
                                n_correct=n_full_correct,
                            )
                            val = e_res.get("energy_j_per_decision")
                            if val is not None and not np.isnan(val):
                                energy_per_dec_str = f"{val:.4f}"
                            break

            out_rows.append({
                "model": m,
                "condition": c,
                "primary_block": loaded.primary_block_by_cell.get(key) or "None",
                "n_attempts": n_attempts,
                "n_valid": n_valid,
                "schema_valid_rate": schema_valid_rate,
                "unsafe_policy_rate": unsafe_rate,
                "actuated_vector_match": actuated_vector_match,
                "full_policy_match": full_policy_match,
                "target_cluster_state_dep_acc": target_cluster_state_dep_acc,
                "action_acc": field_accs["action"],
                "class_acc": field_accs["class"],
                "scope_acc": field_accs["scope"],
                "target_cluster_acc": field_accs["target_cluster"],
                "priority_acc": field_accs["priority"],
                "prb_share_acc": field_accs["prb_share"],
                "edge_site_acc": field_accs["edge_site"],
                "latency_target_acc": field_accs["latency_target"],
                "duration_acc": field_accs["duration"],
                "n_latency": len(lats),
                "latency_p50": p50,
                "latency_p95": p95,
                "latency_p99": p99,
                "net_latency_status": net_status,
                "rtt_probe_median": rtt_median,
                "latency_net_p50": net_p50,
                "latency_net_p95": net_p95,
                "latency_net_p99": net_p99,
                "mean_input_tokens": mean_in,
                "mean_output_tokens": mean_out,
                "n_cost_rows": len(cost_rows),
                "n_correct_cost_rows": n_correct_costed,
                "cost_per_1000_correct_usd": cost_per_1k,
                "energy_j_per_decision": energy_per_dec_str,
            })

    fieldnames = list(out_rows[0].keys()) if out_rows else []
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(out_rows)
    return out_rows, buf.getvalue()


def compute_near_rt_feasibility_table(
    loaded: LoadedRuns,
    expected_models: list[str] = EXPECTED_MODELS,
    expected_conditions: list[str] = EXPECTED_CONDITIONS,
    delta_e2: float = DELTA_E2_S,
    deadline: float = NEAR_RT_DEADLINE_S,
    seed: int = DEFAULT_SEED,
    n_resamples: int = DEFAULT_RESAMPLES,
) -> tuple[list[dict[str, Any]], str]:
    """Compute near-RT feasibility phi = P(latency + delta_e2 <= 1s) with bootstrap CI."""
    out_rows: list[dict[str, Any]] = []

    for m in expected_models:
        for c in expected_conditions:
            key = (m, c)
            p_rows = loaded.primary_by_cell.get(key, [])
            n = len(p_rows)

            if n == 0:
                out_rows.append({
                    "model": m,
                    "condition": c,
                    "n": 0,
                    "n_feasible": 0,
                    "delta_e2_s": delta_e2,
                    "deadline_s": deadline,
                    "phi": float("nan"),
                    "phi_ci_low": float("nan"),
                    "phi_ci_high": float("nan"),
                })
                continue

            # Indicator: latency is reported, no transport error, and latency + delta_e2 <= deadline
            binary = [
                1.0 if (
                    r.get("latency_s") is not None
                    and r.get("error_type") is None
                    and float(r["latency_s"]) + delta_e2 <= deadline
                ) else 0.0
                for r in p_rows
            ]
            n_feas = int(sum(binary))
            phi = float(np.mean(binary))
            ci = bootstrap_ci(binary, seed=seed, n_resamples=n_resamples)

            out_rows.append({
                "model": m,
                "condition": c,
                "n": n,
                "n_feasible": n_feas,
                "delta_e2_s": delta_e2,
                "deadline_s": deadline,
                "phi": phi,
                "phi_ci_low": ci[0],
                "phi_ci_high": ci[1],
            })

    fieldnames = list(out_rows[0].keys()) if out_rows else []
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(out_rows)
    return out_rows, buf.getvalue()


# ---------------------------------------------------------------------------
# H3 Confirmatory Hypothesis & Comparisons
# ---------------------------------------------------------------------------

def holm_fixed_family(raw_p: dict[str, float], family: list[str]) -> dict[str, float]:
    """Holm adjustment over a fixed family; members without a raw p are untested.

    The family never shrinks: an untested member keeps its place with p = 1.0 (the most
    conservative placement for the tested members) and gets NaN back.
    """
    full = {m: (raw_p[m] if m in raw_p else 1.0) for m in family}
    adj = holm_adjust(full)
    return {m: (adj[m] if m in raw_p else float("nan")) for m in family}


def _tc_vector(rows: list[dict[str, Any]], case_ids: list[str]) -> np.ndarray | None:
    """0/1 target_cluster correctness over case_ids, or None unless every case has a row."""
    by_id = {str(r["case_id"]): r for r in rows}
    if not all(cid in by_id for cid in case_ids):
        return None
    return np.array([1.0 if is_scored_correct(by_id[cid], ("target_cluster",)) else 0.0 for cid in case_ids])


def evaluate_h3_confirmatory(
    loaded: LoadedRuns,
    corpus_cases: dict[str, dict[str, CorpusCase]],
    seed: int = DEFAULT_SEED,
    n_resamples: int = DEFAULT_RESAMPLES,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, str]:
    """Evaluate H3 (target_cluster acc at c3_fresh minus c57_fresh > 0, paired case bootstrap).

    The Holm family is fixed to THE_SIX_MODELS; a member without paired primary data on all 150
    shared state-dependent cases is untested and not resolved. REFERENCE_MODELS are reported
    outside the family (no Holm p, no resolution). At c57_fresh: Cochran's Q across exactly the
    six, and McNemar for each other member vs Jev-1.13.0; a test lacking data is "incomplete".
    """
    state_dep_cids = shared_state_dependent_ids(corpus_cases, "c3_fresh", "c57_fresh")
    n_cases = len(state_dep_cids)
    nan = float("nan")
    raw_p_values: dict[str, float] = {}
    h3_records: dict[str, dict[str, Any]] = {}
    c57_vectors: dict[str, np.ndarray | None] = {}

    for m in THE_SIX_MODELS + REFERENCE_MODELS:
        y3 = _tc_vector(loaded.primary_by_cell.get((m, "c3_fresh"), []), state_dep_cids)
        y57 = _tc_vector(loaded.primary_by_cell.get((m, "c57_fresh"), []), state_dep_cids)
        c57_vectors[m] = y57
        if y3 is None or y57 is None:
            continue
        diff = y3 - y57
        point = float(np.mean(diff))
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, n_cases, size=(n_resamples, n_cases))
        samples = np.mean(diff[indices], axis=1)
        ci_low = min(float(np.percentile(samples, 2.5)), point)
        ci_high = max(float(np.percentile(samples, 97.5)), point)
        p_raw = invert_bootstrap_pvalue(samples, null_value=0.0)
        if m in THE_SIX_MODELS:
            raw_p_values[m] = p_raw
        h3_records[m] = {
            "acc_c3_fresh": float(np.mean(y3)),
            "acc_c57_fresh": float(np.mean(y57)),
            "delta": point,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "p_raw": p_raw,
        }

    adj_p_values = holm_fixed_family(raw_p_values, THE_SIX_MODELS)

    h3_rows: list[dict[str, Any]] = []
    for m in THE_SIX_MODELS + REFERENCE_MODELS:
        in_family = m in THE_SIX_MODELS
        rec = h3_records.get(m)
        if rec is None:
            status = "untested_incomplete" if in_family else "reference_incomplete"
        else:
            status = "tested" if in_family else "reference"
        p_holm = adj_p_values[m] if in_family else nan
        if in_family:
            resolved: bool | str = bool(rec is not None and rec["ci_low"] > 0.0 and p_holm < 0.05)
        else:
            resolved = "n/a (reference)"
        h3_rows.append({
            "model": m,
            "family": "H3" if in_family else "reference",
            "holm_family_size": len(THE_SIX_MODELS) if in_family else "",
            "n_cases": n_cases,
            "acc_c3_fresh": rec["acc_c3_fresh"] if rec else nan,
            "acc_c57_fresh": rec["acc_c57_fresh"] if rec else nan,
            "delta_c3_minus_c57": rec["delta"] if rec else nan,
            "ci_low": rec["ci_low"] if rec else nan,
            "ci_high": rec["ci_high"] if rec else nan,
            "p_raw": rec["p_raw"] if rec else nan,
            "p_holm": p_holm,
            "resolved": resolved,
            "status": status,
        })

    # Cochran's Q across exactly the six at c57_fresh
    comp_rows: list[dict[str, Any]] = []
    missing = [m for m in THE_SIX_MODELS if c57_vectors.get(m) is None]
    if not missing:
        matrix = np.column_stack([c57_vectors[m] for m in THE_SIX_MODELS]).astype(int)
        q_res = cochrans_q(matrix)
        q_stat, q_df, q_p, q_status = q_res["q"], q_res["df"], q_res["p"], "tested"
    else:
        q_stat, q_df, q_p, q_status = nan, len(THE_SIX_MODELS) - 1, nan, "incomplete"
    comp_rows.append({
        "test_type": "Cochran_Q",
        "condition": "c57_fresh",
        "model_1": "All six (" + ", ".join(THE_SIX_MODELS) + ")",
        "model_2": "N/A",
        "statistic_name": "Q",
        "statistic_value": q_stat,
        "df": q_df,
        "p_value": q_p,
        "n_cases": n_cases,
        "status": q_status,
        "details": f"k={len(THE_SIX_MODELS)} interpreters"
        + (f"; missing paired data: {', '.join(missing)}" if missing else ""),
    })

    # Paired McNemar: each other member of the six vs Jev-1.13.0
    ref = "Jev-1.13.0"
    for m in THE_SIX_MODELS:
        if m == ref:
            continue
        a, j = c57_vectors.get(m), c57_vectors.get(ref)
        if a is None or j is None:
            lacking = ", ".join(x for x, v in ((m, a), (ref, j)) if v is None)
            comp_rows.append({
                "test_type": "McNemar", "condition": "c57_fresh", "model_1": m, "model_2": ref,
                "statistic_name": "b/c", "statistic_value": "", "df": 1, "p_value": nan,
                "n_cases": n_cases, "status": "incomplete",
                "details": f"missing paired data: {lacking}",
            })
            continue
        b_disc = int(np.sum((a == 1) & (j == 0)))
        c_disc = int(np.sum((a == 0) & (j == 1)))
        comp_rows.append({
            "test_type": "McNemar",
            "condition": "c57_fresh",
            "model_1": m,
            "model_2": ref,
            "statistic_name": "b/c",
            "statistic_value": f"{b_disc}/{c_disc}",
            "df": 1,
            "p_value": exact_mcnemar_test(a.astype(bool), j.astype(bool)),
            "n_cases": n_cases,
            "status": "tested",
            "details": f"{m} correct only: {b_disc}, Jev correct only: {c_disc}",
        })

    h3_buf = io.StringIO()
    h3_writer = csv.DictWriter(h3_buf, fieldnames=list(h3_rows[0].keys()))
    h3_writer.writeheader()
    h3_writer.writerows(h3_rows)

    comp_buf = io.StringIO()
    comp_writer = csv.DictWriter(comp_buf, fieldnames=list(comp_rows[0].keys()))
    comp_writer.writeheader()
    comp_writer.writerows(comp_rows)

    return h3_rows, comp_rows, h3_buf.getvalue(), comp_buf.getvalue()


# ---------------------------------------------------------------------------
# Exploratory Analysis: Quality Contrasts & Cell Count Curves
# ---------------------------------------------------------------------------

def compute_exploratory_contrasts(
    loaded: LoadedRuns,
    corpus_cases: dict[str, dict[str, CorpusCase]],
    expected_models: list[str] = EXPECTED_MODELS,
    seed: int = DEFAULT_SEED,
    n_resamples: int = DEFAULT_RESAMPLES,
) -> tuple[list[dict[str, Any]], str]:
    """Compute c3 vs c57 accuracy contrasts on state-dependent cases for stale/noisy/contradictory."""
    out_rows: list[dict[str, Any]] = []

    for q in ["stale", "noisy", "contradictory"]:
        c3_cond = f"c3_{q}"
        c57_cond = f"c57_{q}"

        state_dep_cids = shared_state_dependent_ids(corpus_cases, c3_cond, c57_cond)

        for m in expected_models:
            y3 = _tc_vector(loaded.primary_by_cell.get((m, c3_cond), []), state_dep_cids)
            y57 = _tc_vector(loaded.primary_by_cell.get((m, c57_cond), []), state_dep_cids)

            if y3 is None or y57 is None:
                out_rows.append({
                    "quality": q,
                    "model": m,
                    "n_cases": len(state_dep_cids),
                    "acc_c3": float("nan"),
                    "acc_c57": float("nan"),
                    "delta_c3_minus_c57": float("nan"),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                    "p_value": float("nan"),
                    "status": "missing_or_incomplete",
                })
                continue

            acc3 = float(np.mean(y3))
            acc57 = float(np.mean(y57))
            diff = y3 - y57
            point = float(np.mean(diff))

            rng = np.random.default_rng(seed)
            n_cases = len(state_dep_cids)
            indices = rng.integers(0, n_cases, size=(n_resamples, n_cases))
            samples = np.mean(diff[indices], axis=1)

            ci_low = float(np.percentile(samples, 2.5))
            ci_high = float(np.percentile(samples, 97.5))
            ci_low = min(ci_low, point)
            ci_high = max(ci_high, point)
            p_val = invert_bootstrap_pvalue(samples, null_value=0.0)

            out_rows.append({
                "quality": q,
                "model": m,
                "n_cases": n_cases,
                "acc_c3": acc3,
                "acc_c57": acc57,
                "delta_c3_minus_c57": point,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_value": p_val,
                "status": "tested",
            })

    fieldnames = list(out_rows[0].keys()) if out_rows else []
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(out_rows)
    return out_rows, buf.getvalue()


def compute_cell_count_curves(
    loaded: LoadedRuns,
    corpus_cases: dict[str, dict[str, CorpusCase]],
    expected_models: list[str] = EXPECTED_MODELS,
) -> tuple[list[dict[str, Any]], str]:
    """Compute accuracy vs cell count (3, 7, 21, 57 cells) curves across all qualities."""
    out_rows: list[dict[str, Any]] = []

    for m in expected_models:
        for q in QUALITIES:
            for n_cells in CELL_COUNTS:
                cond = f"c{n_cells}_{q}"
                p_rows = loaded.primary_by_cell.get((m, cond), [])

                sd_cids = sorted(cid for cid, cc in corpus_cases.get(cond, {}).items() if cc.is_state_dependent)
                p_rows_map = {str(r["case_id"]): r for r in p_rows}

                if not p_rows:
                    out_rows.append({
                        "model": m,
                        "quality": q,
                        "n_cells": n_cells,
                        "condition": cond,
                        "target_cluster_state_dep_acc": float("nan"),
                        "actuated_vector_match": float("nan"),
                        "full_policy_match": float("nan"),
                    })
                    continue

                sd_acc = float(np.mean([
                    1.0 if (cid in p_rows_map and is_scored_correct(p_rows_map[cid], ("target_cluster",))) else 0.0
                    for cid in sd_cids
                ])) if sd_cids else float("nan")

                # Over all primary attempts; invalid / transport-failed attempts score false
                act_match = float(np.mean([is_scored_correct(r, ACTUATED_FIELDS) for r in p_rows]))
                em_match = float(np.mean([is_full_policy_correct(r) for r in p_rows]))

                out_rows.append({
                    "model": m,
                    "quality": q,
                    "n_cells": n_cells,
                    "condition": cond,
                    "target_cluster_state_dep_acc": sd_acc,
                    "actuated_vector_match": act_match,
                    "full_policy_match": em_match,
                })

    fieldnames = list(out_rows[0].keys()) if out_rows else []
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(out_rows)
    return out_rows, buf.getvalue()


# ---------------------------------------------------------------------------
# Sensitivity Table
# ---------------------------------------------------------------------------

def compute_sensitivity_table(
    loaded: LoadedRuns,
    corpus_cases: dict[str, dict[str, CorpusCase]],
    seed: int = DEFAULT_SEED,
) -> tuple[list[dict[str, Any]], str]:
    """Compare primary vs sensitivity blocks for each affected (model, condition).

    D-3 per-case cells are left out: their other attempts are re-issues of the same cases, reported
    in the availability table.
    """
    out_rows: list[dict[str, Any]] = []

    for (m, c), s_rows in sorted(loaded.sensitivity_by_cell.items()):
        if not s_rows or (m, c) in loaded.per_case_cells:
            continue
        p_rows = loaded.primary_by_cell.get((m, c), [])
        s_block_names = sorted(list({r.get("_block_name", "sensitivity") for r in s_rows}))

        for s_bname in s_block_names:
            b_s_rows = [r for r in s_rows if r.get("_block_name") == s_bname]
            p_bname = loaded.primary_block_by_cell.get((m, c)) or "none"

            # Compare metrics
            p_n = len(p_rows)
            s_n = len(b_s_rows)

            p_valid = [r for r in p_rows if bool(r.get("valid")) and r.get("error_type") is None]
            s_valid = [r for r in b_s_rows if bool(r.get("valid")) and r.get("error_type") is None]

            p_err = sum(1 for r in p_rows if r.get("error_type") is not None)
            s_err = sum(1 for r in b_s_rows if r.get("error_type") is not None)

            p_avail = (1.0 - p_err / p_n) if p_n > 0 else float("nan")
            s_avail = (1.0 - s_err / s_n) if s_n > 0 else float("nan")

            # Same scoring as the quality table: correctness over all attempts, latency over all
            # rows that report it
            p_em = float(np.mean([is_full_policy_correct(r) for r in p_rows])) if p_rows else float("nan")
            s_em = float(np.mean([is_full_policy_correct(r) for r in b_s_rows])) if b_s_rows else float("nan")
            p_act = float(np.mean([is_scored_correct(r, ACTUATED_FIELDS) for r in p_rows])) if p_rows else float("nan")
            s_act = float(np.mean([is_scored_correct(r, ACTUATED_FIELDS) for r in b_s_rows])) if b_s_rows else float("nan")

            p_lats = [float(r["latency_s"]) for r in p_rows if r.get("latency_s") is not None]
            s_lats = [float(r["latency_s"]) for r in b_s_rows if r.get("latency_s") is not None]

            p_p50 = float(np.percentile(p_lats, 50)) if p_lats else float("nan")
            s_p50 = float(np.percentile(s_lats, 50)) if s_lats else float("nan")
            p_p95 = float(np.percentile(p_lats, 95)) if p_lats else float("nan")
            s_p95 = float(np.percentile(s_lats, 95)) if s_lats else float("nan")

            metrics = [
                ("total_cases", "Total Cases (n)", p_n, s_n),
                ("valid_cases", "Valid Cases", len(p_valid), len(s_valid)),
                ("transport_errors", "Transport Errors", p_err, s_err),
                ("availability_rate", "Availability Rate", p_avail, s_avail),
                ("actuated_vector_match", "Actuated Vector Match", p_act, s_act),
                ("full_policy_match", "Full Policy Match (EM)", p_em, s_em),
                ("latency_p50", "Latency p50 (s)", p_p50, s_p50),
                ("latency_p95", "Latency p95 (s)", p_p95, s_p95),
            ]

            for m_id, m_name, p_val, s_val in metrics:
                delta_str = ""
                if isinstance(p_val, (int, float)) and isinstance(s_val, (int, float)) and not (np.isnan(p_val) or np.isnan(s_val)):
                    delta_str = f"{s_val - p_val:+.4f}"
                out_rows.append({
                    "model": m,
                    "condition": c,
                    "primary_source": p_bname,
                    "sensitivity_source": s_bname,
                    "metric_id": m_id,
                    "metric_name": m_name,
                    "primary_value": f"{p_val:.4f}" if isinstance(p_val, float) and not np.isnan(p_val) else str(p_val),
                    "sensitivity_value": f"{s_val:.4f}" if isinstance(s_val, float) and not np.isnan(s_val) else str(s_val),
                    "delta": delta_str,
                })

    fieldnames = [
        "model", "condition", "primary_source", "sensitivity_source",
        "metric_id", "metric_name", "primary_value", "sensitivity_value", "delta",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(out_rows)
    return out_rows, buf.getvalue()


# ---------------------------------------------------------------------------
# Results.md Generation (Deterministic & Traceable)
# ---------------------------------------------------------------------------

def generate_results_md(
    avail_rows: list[dict[str, Any]],
    qc_rows: list[dict[str, Any]],
    near_rt_rows: list[dict[str, Any]],
    h3_rows: list[dict[str, Any]],
    comp_rows: list[dict[str, Any]],
    exploratory_rows: list[dict[str, Any]],
    curves_rows: list[dict[str, Any]],
    sensitivity_rows: list[dict[str, Any]],
    seed: int = DEFAULT_SEED,
    n_resamples: int = DEFAULT_RESAMPLES,
) -> str:
    """Generate deterministic results.md with complete traceability to CSV files."""
    pre_registered = seed == DEFAULT_SEED and n_resamples == DEFAULT_RESAMPLES
    boot_str = f"seed {seed}, {n_resamples:,} bootstrap resamples"
    if pre_registered:
        boot_line = f"Bootstrap: {boot_str} (pre-registered)."
    else:
        boot_line = (
            f"Bootstrap: {boot_str} -- SENSITIVITY RUN: differs from the pre-registered "
            f"seed {DEFAULT_SEED}, {DEFAULT_RESAMPLES:,} resamples; not the confirmatory result."
        )
    lines: list[str] = [
        "# EXP-2026-003 C1 Analysis Results (Frozen v0.2)",
        "",
        "This report summarizes intent interpretation quality, near-RT feasibility ($\\phi$),",
        "confirmatory hypothesis H3, telemetry exploratory contrasts, and sensitivity blocks.",
        "All numbers are strictly deterministic and traceable to generated CSV files.",
        "",
        boot_line,
        "",
        "---",
        "",
        "## 1. Roster Coverage & Availability",
        "",
        "Traceable to `availability.csv` (per-block attempts and HTTP 429 rates in `block_attempts`).",
        "Qwen3.8-Flash (D-3): primary row per case = first transport-error-free attempt across its blocks.",
        "",
        "| Interpreter | Condition | Status | Primary Block | Primary Calls | Primary Errors | Error Types | Primary Avail | Error >5% | Blocks >5% Error | Sensitivity Blocks | Total Calls | Total Avail | HTTP 429 (all attempts) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for r in avail_rows:
        p_avail_str = f"{r['primary_avail_rate']:.4f}" if not np.isnan(r['primary_avail_rate']) else "N/A"
        tot_avail_str = f"{r['total_avail_rate']:.4f}" if not np.isnan(r['total_avail_rate']) else "N/A"
        err_flag = "**YES**" if r["error_gt_5pct"] else "no"
        http_429_str = f"{r['total_http_429']} ({r['total_http_429_rate']:.1%})" if r["total_n"] else "N/A"
        lines.append(
            f"| {r['model']} | {r['condition']} | {r['status']} | {r['primary_block']} | {r['primary_n']} | {r['primary_errors']} | "
            f"{r['primary_error_types']} | {p_avail_str} | {err_flag} | {r['blocks_error_gt_5pct']} | {r['sensitivity_block_names']} | {r['total_n']} | {tot_avail_str} | "
            f"{http_429_str} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Near-RT Feasibility ($\\phi = P(\\text{latency} + \\delta_{E2} \\le 1\\text{ s})$)",
        "",
        f"Traceable to `near_rt_feasibility.csv` ($\\delta_{{E2}} = 0.005\\text{{ s}}$, {boot_str}).",
        "",
        "| Interpreter | Condition | Total (n) | Feasible | $\\phi$ | 95% Bootstrap CI |",
        "|---|---|---|---|---|---|",
    ])

    for r in near_rt_rows:
        if r["n"] > 0:
            phi_str = f"{r['phi']:.4f}"
            ci_str = f"[{r['phi_ci_low']:.4f}, {r['phi_ci_high']:.4f}]"
        else:
            phi_str = "N/A"
            ci_str = "N/A"
        lines.append(
            f"| {r['model']} | {r['condition']} | {r['n']} | {r['n_feasible']} | {phi_str} | {ci_str} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Confirmatory Hypothesis H3 (Telemetry Grounding Contrast)",
        "",
        "Traceable to `h3_confirmatory.csv` and `h3_c57_comparisons.csv`.",
        "H3 tests target_cluster accuracy at `c3_fresh` minus `c57_fresh` > 0 over the 150 state-dependent cases.",
        "\"Resolved\" requires 95% CI to exclude 0 and Holm-adjusted p < 0.05.",
        f"Holm family: the six interpreters ({', '.join(THE_SIX_MODELS)}); an untested member stays in the",
        "family (p = 1 in the adjustment) and is not resolved. Reference models are reported outside the family.",
        f"{boot_str}.",
        "",
        "### H3 Paired Contrast",
        "",
        "| Interpreter | Cases | Acc (c3_fresh) | Acc (c57_fresh) | $\\Delta$ (c3 - c57) | 95% Bootstrap CI | Raw p | Holm p | Resolved |",
        "|---|---|---|---|---|---|---|---|---|",
    ])

    for r in h3_rows:
        if r["status"] in ("tested", "reference"):
            delta_str = f"{r['delta_c3_minus_c57']:+.4f}"
            ci_str = f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
            p_raw_str = f"{r['p_raw']:.4e}"
            if r["status"] == "tested":
                p_holm_str = f"{r['p_holm']:.4e}"
                res_str = "**TRUE**" if r["resolved"] else "FALSE"
            else:
                p_holm_str = "N/A (outside family)"
                res_str = "reference"
            lines.append(
                f"| {r['model']} | {r['n_cases']} | {r['acc_c3_fresh']:.4f} | {r['acc_c57_fresh']:.4f} | "
                f"{delta_str} | {ci_str} | {p_raw_str} | {p_holm_str} | {res_str} |"
            )
        else:
            res_str = "untested (incomplete; not resolved)" if r["family"] == "H3" else "reference (incomplete)"
            lines.append(
                f"| {r['model']} | {r['n_cases']} | N/A | N/A | N/A | N/A | N/A | N/A | {res_str} |"
            )

    lines.extend([
        "",
        "### Multi-Interpreter Comparisons at `c57_fresh`",
        "",
        "| Test Type | Condition | Model 1 | Model 2 | Statistic | Value | df | p-value | Status | Details |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ])

    for r in comp_rows:
        if r["status"] == "tested":
            val_str = f"{r['statistic_value']:.4f}" if isinstance(r['statistic_value'], float) else str(r['statistic_value'])
            p_str = f"{r['p_value']:.4e}"
        else:
            val_str = p_str = "N/A"
        lines.append(
            f"| {r['test_type']} | {r['condition']} | {r['model_1']} | {r['model_2']} | {r['statistic_name']} | "
            f"{val_str} | {r['df']} | {p_str} | {r['status']} | {r['details']} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Quality, Accuracy & Cost Summary (Selected Key Metrics)",
        "",
        "Traceable to `quality_and_cost.csv`. Correctness is over all primary attempts (schema-invalid and",
        "transport-failed attempts score false); latency and tokens over every row that reports them; fees per",
        "1,000 correct over the cost-reporting rows (numerator and denominator alike).",
        "",
        "| Interpreter | Condition | Attempts | Valid n | Schema Valid Rate | Actuated Vector Match | Full Policy Match | Target Cluster (State-Dep) | Latency p50 (s) | Net p50 (s) | Cost / 1k Correct ($) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ])

    def fmt(v: float, spec: str, prefix: str = "") -> str:
        return "N/A" if v is None or np.isnan(v) else f"{prefix}{v:{spec}}"

    for r in qc_rows:
        if r["net_latency_status"] == "ok":
            net_str = fmt(r["latency_net_p50"], ".3f")
        elif r["net_latency_status"] == "inapplicable_selfhosted":
            net_str = "n/a (self-hosted)"
        elif r["net_latency_status"] == "no_probe":
            net_str = "unavailable (no probe)"
        else:
            net_str = "N/A"
        lines.append(
            f"| {r['model']} | {r['condition']} | {r['n_attempts']} | {r['n_valid']} | {fmt(r['schema_valid_rate'], '.4f')} | "
            f"{fmt(r['actuated_vector_match'], '.4f')} | {fmt(r['full_policy_match'], '.4f')} | "
            f"{fmt(r['target_cluster_state_dep_acc'], '.4f')} | {fmt(r['latency_p50'], '.3f')} | {net_str} | "
            f"{fmt(r['cost_per_1000_correct_usd'], '.4f', '$')} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 5. Exploratory Quality Contrasts (c3 minus c57)",
        "",
        f"Traceable to `exploratory_contrasts.csv` ({boot_str}).",
        "",
        "| Quality | Interpreter | Cases | Acc (c3) | Acc (c57) | $\\Delta$ (c3 - c57) | 95% Bootstrap CI | p-value | Status |",
        "|---|---|---|---|---|---|---|---|---|",
    ])

    for r in exploratory_rows:
        if r["status"] == "tested":
            lines.append(
                f"| {r['quality']} | {r['model']} | {r['n_cases']} | {r['acc_c3']:.4f} | {r['acc_c57']:.4f} | "
                f"{r['delta_c3_minus_c57']:+.4f} | [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | {r['p_value']:.4e} | tested |"
            )
        else:
            lines.append(
                f"| {r['quality']} | {r['model']} | {r['n_cases']} | N/A | N/A | N/A | N/A | N/A | Incomplete |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## 6. Sensitivity Re-Run Comparisons",
        "",
        "Traceable to `sensitivity.csv`.",
        "",
        "| Interpreter | Condition | Primary Source | Sensitivity Source | Metric | Primary | Sensitivity | $\\Delta$ |",
        "|---|---|---|---|---|---|---|---|",
    ])

    if sensitivity_rows:
        for r in sensitivity_rows:
            lines.append(
                f"| {r['model']} | {r['condition']} | {r['primary_source']} | {r['sensitivity_source']} | "
                f"{r['metric_name']} | {r['primary_value']} | {r['sensitivity_value']} | {r['delta']} |"
            )
    else:
        lines.append("| None | None | None | None | None | None | None | None |")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Master Pipeline Runner
# ---------------------------------------------------------------------------

def run_c1_analysis(
    run_dirs: list[Path | str],
    corpus_dir: Path | str,
    out_dir: Path | str,
    seed: int = DEFAULT_SEED,
    n_resamples: int = DEFAULT_RESAMPLES,
) -> dict[str, Any]:
    """Execute complete C1 analysis pipeline and export deterministic CSVs and results.md."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Discover run files
    files = find_run_files(run_dirs)
    if not files["ledgers"]:
        raise FileNotFoundError(f"No ledger.jsonl files found in run dirs: {run_dirs}")

    # 2. Load corpus; every condition must be present (it defines the case sets)
    corpus_cases = load_corpus_cases(corpus_dir)
    missing_conds = [c for c in EXPECTED_CONDITIONS if not corpus_cases.get(c)]
    if missing_conds:
        raise ValueError(f"Corpus validation failed: no test cases for {', '.join(missing_conds)}")
    expected_case_ids = {c: set(corpus_cases[c]) for c in EXPECTED_CONDITIONS}

    # 3. Load probes (per block, model, condition)
    probe_medians = load_probes(files["probes"], files["block_of"])

    # 4. Load ledgers; one primary block per (model, condition)
    loaded = load_all_c1_runs(files["ledgers"], files["block_of"], expected_case_ids)

    # 5. Compute tables
    avail_rows, avail_csv = compute_availability_table(loaded)
    (out_path / "availability.csv").write_text(avail_csv, encoding="utf-8")

    qc_rows, qc_csv = compute_quality_and_cost_table(
        loaded, corpus_cases, probe_medians, files["power_traces"]
    )
    (out_path / "quality_and_cost.csv").write_text(qc_csv, encoding="utf-8")

    near_rt_rows, near_rt_csv = compute_near_rt_feasibility_table(
        loaded, seed=seed, n_resamples=n_resamples
    )
    (out_path / "near_rt_feasibility.csv").write_text(near_rt_csv, encoding="utf-8")

    h3_rows, comp_rows, h3_csv, comp_csv = evaluate_h3_confirmatory(
        loaded, corpus_cases, seed=seed, n_resamples=n_resamples
    )
    (out_path / "h3_confirmatory.csv").write_text(h3_csv, encoding="utf-8")
    (out_path / "h3_c57_comparisons.csv").write_text(comp_csv, encoding="utf-8")

    exp_rows, exp_csv = compute_exploratory_contrasts(
        loaded, corpus_cases, seed=seed, n_resamples=n_resamples
    )
    (out_path / "exploratory_contrasts.csv").write_text(exp_csv, encoding="utf-8")

    curves_rows, curves_csv = compute_cell_count_curves(loaded, corpus_cases)
    (out_path / "cell_count_curves.csv").write_text(curves_csv, encoding="utf-8")

    sens_rows, sens_csv = compute_sensitivity_table(loaded, corpus_cases, seed=seed)
    (out_path / "sensitivity.csv").write_text(sens_csv, encoding="utf-8")

    # 6. Generate results.md
    results_md = generate_results_md(
        avail_rows, qc_rows, near_rt_rows, h3_rows, comp_rows, exp_rows, curves_rows, sens_rows,
        seed=seed, n_resamples=n_resamples,
    )
    (out_path / "results.md").write_text(results_md, encoding="utf-8")

    return {
        "files_written": [
            "availability.csv",
            "quality_and_cost.csv",
            "near_rt_feasibility.csv",
            "h3_confirmatory.csv",
            "h3_c57_comparisons.csv",
            "exploratory_contrasts.csv",
            "cell_count_curves.csv",
            "sensitivity.csv",
            "results.md",
        ],
        "avail_rows": avail_rows,
        "qc_rows": qc_rows,
        "near_rt_rows": near_rt_rows,
        "h3_rows": h3_rows,
        "comp_rows": comp_rows,
        "exploratory_rows": exp_rows,
        "curves_rows": curves_rows,
        "sensitivity_rows": sens_rows,
        "seed": seed,
        "n_resamples": n_resamples,
        "pre_registered_bootstrap": seed == DEFAULT_SEED and n_resamples == DEFAULT_RESAMPLES,
    }
