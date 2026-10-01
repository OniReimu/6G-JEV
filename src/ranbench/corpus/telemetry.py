"""KPM telemetry tables for RANIntent v1 (RQ4): construction, rendering, parsing and the reference reader.

Per tuple, a 57-cell universe fixes the cluster of every cell and the two extreme cells: the highest
prb_util_pct and the lowest edge_ue_thr_mbps, in different clusters. For a state-dependent tuple the intent
metric's extreme lies in target_cluster; every other cell trails it by at least the metric margin, so the answer is
the same at every cell count. A table of n cells always holds both extreme cells (and, for a named-scope tuple, a
cell of its scope cluster). The quality transform then acts on that base table:
  fresh          - the base table;
  stale          - base rows get measured_at within fresh_age_s; extra rows older than stale_after_s are added;
  noisy          - bounded noise per metric (margin > 2 x bound, so the extreme cell never changes);
  contradictory  - base rows are E2_KPM; extra O1_PM rows that disagree are added.
Extra rows go to round(extra_row_share x n) cells. With probability decoy_probability one extra (stale / O1) row is a
decoy: a cell outside the intent-metric extreme's cluster whose value beats the true extreme, so a reading that
ignores the rule gives a wrong cluster. The config sets extra_row_share = 1.0 (every cell gets exactly one extra row):
with fewer extra rows, which cells repeat would leak the answer, because the decoy is never in the target cluster.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import random
import re
from typing import Any

from src.edgebench.corpus.pad import get_tokenizer
from src.edgebench.corpus.tuples import TupleItem, _stable_seed
from src.ranbench.corpus.spec import CLUSTERS, METRIC_COLUMNS, METRIC_DECIMALS, SCOPE_METRIC

TIME_FMT = "%Y-%m-%dT%H:%M:%SZ"
_HEADER_RE = re.compile(r"^KPM telemetry snapshot at (\S+); (\d+) cells$")
_NOISE_PREFIX = "Noise bounds: "


def _u(rng: random.Random, lo: float, hi: float, dec: int) -> float:
    return round(rng.uniform(lo, hi), dec)


def _ordinary_values(rng: random.Random, top: float, bottom: float, cfg: dict[str, Any]) -> dict[str, float]:
    """Metric values of a non-extreme cell: below top - margin (PRB) and above bottom + margin (edge)."""
    t = cfg["telemetry"]
    prb = _u(rng, t["prb_util_pct"]["others_min"], top - t["prb_util_pct"]["margin"], 1)
    d = t["p95_delay_ms"]
    return {
        "prb_util_pct": prb,
        "dl_thr_mbps": _u(rng, *t["dl_thr_mbps"]["range"], 1),
        "p95_delay_ms": round(d["base"] + d["per_prb_pct"] * prb + rng.uniform(*d["jitter"]), 1),
        "edge_ue_thr_mbps": _u(rng, bottom + t["edge_ue_thr_mbps"]["margin"], t["edge_ue_thr_mbps"]["others_max"], 2),
    }


def intent_metric(item: TupleItem) -> tuple[str, str]:
    """(metric, max|min) that decides the answer; named-scope tuples use PRB utilisation for their decoys."""
    return SCOPE_METRIC.get(item.tuple_labels["scope"], ("prb_util_pct", "max"))


def build_universe(item: TupleItem, cfg: dict[str, Any]) -> dict[str, Any]:
    """The per-tuple 57-cell universe (independent of cell count and quality)."""
    t = cfg["telemetry"]
    n_max = max(cfg["cell_counts"])
    rng = random.Random(_stable_seed(f"{item.tuple_id}:universe", cfg["base_seed"]))
    ids = [f"cell-{k}" for k in sorted(rng.sample(range(100, 1000), n_max))]
    order = list(CLUSTERS)
    rng.shuffle(order)
    labels = [c for i, c in enumerate(order) for _ in range(n_max // 4 + (1 if i < n_max % 4 else 0))]
    rng.shuffle(labels)
    cluster_of = dict(zip(ids, labels))

    lab = item.tuple_labels
    if lab["scope"] in SCOPE_METRIC:
        target = lab["target_cluster"]
        other = rng.choice([c for c in CLUSTERS if c != target])
        if SCOPE_METRIC[lab["scope"]][0] == "prb_util_pct":
            prb_cluster, edge_cluster = target, other
        else:
            prb_cluster, edge_cluster = other, target
        must = target
    else:
        prb_cluster = rng.choice(CLUSTERS)
        edge_cluster = rng.choice([c for c in CLUSTERS if c != prb_cluster])
        must = lab["scope"] if lab["scope"] in CLUSTERS else prb_cluster
    top_cell = rng.choice([c for c in ids if cluster_of[c] == prb_cluster])
    bottom_cell = rng.choice([c for c in ids if cluster_of[c] == edge_cluster])

    top = _u(rng, *t["prb_util_pct"]["extreme"], 1)
    bottom = _u(rng, *t["edge_ue_thr_mbps"]["extreme"], 2)
    cells: dict[str, dict[str, Any]] = {}
    for cid in ids:
        vals = _ordinary_values(rng, top, bottom, cfg)
        if cid == top_cell:
            d = t["p95_delay_ms"]
            vals["prb_util_pct"] = top
            vals["p95_delay_ms"] = round(d["base"] + d["per_prb_pct"] * top + rng.uniform(*d["jitter"]), 1)
        if cid == bottom_cell:
            vals["edge_ue_thr_mbps"] = bottom
        cells[cid] = {"cell_id": cid, "cluster": cluster_of[cid], **vals}
    snapshot = datetime(2026, 10, 1, tzinfo=timezone.utc) + timedelta(
        days=rng.randrange(28), seconds=rng.randrange(6 * 3600, 22 * 3600)
    )
    return {
        "cells": cells,
        "top": top,
        "bottom": bottom,
        "extreme_cells": {"prb_util_pct": top_cell, "edge_ue_thr_mbps": bottom_cell},
        "must_clusters": sorted({prb_cluster, edge_cluster, must}),
        "snapshot": snapshot,
    }


def _select_cells(u: dict[str, Any], n: int, rng: random.Random) -> list[dict[str, Any]]:
    """n cells with near-equal clusters, holding both extreme cells and a cell of every must cluster."""
    cells = u["cells"]
    if n >= len(cells):
        return [cells[c] for c in sorted(cells)]
    base, rem = divmod(n, 4)
    must = list(u["must_clusters"])
    rest = [c for c in CLUSTERS if c not in must]
    rng.shuffle(rest)
    if base == 0:
        if len(must) > rem:
            raise ValueError(f"{n} cells cannot hold the {len(must)} required clusters")
        extra = must + rest[: rem - len(must)]
    else:
        pool = list(CLUSTERS)
        rng.shuffle(pool)
        extra = pool[:rem]
    chosen: list[str] = []
    fixed = set(u["extreme_cells"].values())
    for cl in CLUSTERS:
        quota = base + (1 if cl in extra else 0)
        members = sorted(c for c in cells if cells[c]["cluster"] == cl)
        keep = [c for c in members if c in fixed]
        others = [c for c in members if c not in fixed]
        rng.shuffle(others)
        chosen += (keep + others)[:quota] if len(keep) <= quota else keep
    if len(chosen) != n:
        raise ValueError(f"selected {len(chosen)} cells, expected {n}")
    return [cells[c] for c in sorted(chosen)]


def build_table(item: TupleItem, n_cells: int, quality: str, cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """(table, info) for one (tuple, cell count, quality); info (extreme and decoy cells) is never shown."""
    t = cfg["telemetry"]
    seed = cfg["base_seed"]
    u = build_universe(item, cfg)
    cells = _select_cells(u, n_cells, random.Random(_stable_seed(f"{item.tuple_id}:{n_cells}:select", seed)))
    rng = random.Random(_stable_seed(f"{item.tuple_id}:{n_cells}:{quality}", seed))
    snap = u["snapshot"]
    metric, direction = intent_metric(item)
    extreme_cluster = u["cells"][u["extreme_cells"][metric]]["cluster"]

    columns = ["cell_id", "cluster"]
    if quality == "stale":
        columns.append("measured_at")
    if quality == "contradictory":
        columns.append("source")
    columns += METRIC_COLUMNS

    def row(cell: dict[str, Any], vals: dict[str, float], extra: dict[str, str]) -> dict[str, Any]:
        return {"cell_id": cell["cell_id"], "cluster": cell["cluster"], **extra, **vals}

    base_vals = {c["cell_id"]: {m: c[m] for m in METRIC_COLUMNS} for c in cells}
    rows: list[dict[str, Any]] = []
    noise_bounds = None
    decoy_cell = None
    if quality == "fresh":
        rows = [row(c, base_vals[c["cell_id"]], {}) for c in cells]
    elif quality == "noisy":
        noise_bounds = {m: t[m]["noise"] for m in METRIC_COLUMNS}
        for c in cells:
            vals = {}
            for m in METRIC_COLUMNS:
                v = base_vals[c["cell_id"]][m] + rng.uniform(-noise_bounds[m], noise_bounds[m])
                v = min(v, 100.0) if m == "prb_util_pct" else v
                vals[m] = round(max(v, 0.01 if m == "edge_ue_thr_mbps" else 0.1), METRIC_DECIMALS[m])
            rows.append(row(c, vals, {}))
    elif quality in ("stale", "contradictory"):
        if quality == "stale":
            fresh_ts = {c["cell_id"]: snap - timedelta(seconds=rng.randint(*t["fresh_age_s"])) for c in cells}
            rows = [row(c, base_vals[c["cell_id"]], {"measured_at": fresh_ts[c["cell_id"]].strftime(TIME_FMT)}) for c in cells]
        else:
            rows = [row(c, base_vals[c["cell_id"]], {"source": "E2_KPM"}) for c in cells]
        k = max(1, round(t["extra_row_share"] * n_cells))
        if rng.random() < t["decoy_probability"]:
            decoy = rng.choice([c for c in cells if c["cluster"] != extreme_cluster])
            decoy_cell = decoy["cell_id"]
            extra_cells = [decoy] + rng.sample([c for c in cells if c is not decoy], k - 1)
        else:
            extra_cells = rng.sample(cells, k)
        for c in extra_cells:
            vals = _ordinary_values(rng, u["top"], u["bottom"], cfg)
            if c["cell_id"] == decoy_cell:
                vals[metric] = (
                    _u(rng, u["top"] + 1.5, u["top"] + 3.5, 1) if direction == "max"
                    else _u(rng, 0.05, u["bottom"] - 0.2, 2)
                )
            if quality == "stale":
                ts = snap - timedelta(seconds=rng.randint(*t["stale_age_s"]))
                rows.append(row(c, vals, {"measured_at": ts.strftime(TIME_FMT)}))
            else:
                for m in METRIC_COLUMNS:  # an O1 PM row disagrees with the E2 KPM row on every metric
                    if vals[m] == base_vals[c["cell_id"]][m]:
                        vals[m] = round(vals[m] + 10 ** -METRIC_DECIMALS[m], METRIC_DECIMALS[m])
                rows.append(row(c, vals, {"source": "O1_PM"}))
    else:
        raise ValueError(f"unknown quality {quality!r}")

    keyed = [(r["cell_id"], rng.random(), r) for r in rows]
    keyed.sort(key=lambda x: (x[0], x[1]))
    table = {
        "snapshot_time": snap.strftime(TIME_FMT),
        "n_cells": n_cells,
        "columns": columns,
        "noise_bounds": noise_bounds,
        "rows": [[r[c] for c in columns] for _, _, r in keyed],
    }
    info = {"extreme_cells": dict(u["extreme_cells"]), "decoy_cell": decoy_cell}
    return table, info


def render_table(table: dict[str, Any]) -> str:
    """The telemetry text given to interpreters and the verifier (header lines + CSV)."""
    lines = [f"KPM telemetry snapshot at {table['snapshot_time']}; {table['n_cells']} cells"]
    if table["noise_bounds"]:
        lines.append(_NOISE_PREFIX + ", ".join(f"{m} ±{b}" for m, b in table["noise_bounds"].items()))
    cols = table["columns"]
    lines.append(",".join(cols))
    for r in table["rows"]:
        lines.append(",".join(f"{v:.{METRIC_DECIMALS[c]}f}" if c in METRIC_DECIMALS else str(v) for c, v in zip(cols, r)))
    return "\n".join(lines) + "\n"


def parse_table_text(text: str) -> dict[str, Any]:
    """Inverse of render_table."""
    lines = [l for l in text.splitlines() if l.strip()]
    m = _HEADER_RE.match(lines[0])
    if not m:
        raise ValueError(f"not a telemetry header: {lines[0]!r}")
    i = 1
    noise_bounds = None
    if lines[i].startswith(_NOISE_PREFIX):
        noise_bounds = {}
        for part in lines[i][len(_NOISE_PREFIX):].split(", "):
            name, bound = part.split(" ±")
            noise_bounds[name] = float(bound)
        i += 1
    cols = lines[i].split(",")
    rows = []
    for l in lines[i + 1:]:
        vals = l.split(",")
        rows.append([float(v) if c in METRIC_DECIMALS else v for c, v in zip(cols, vals)])
    return {"snapshot_time": m.group(1), "n_cells": int(m.group(2)), "columns": cols, "noise_bounds": noise_bounds, "rows": rows}


def resolve_extreme(
    table: dict[str, Any], metric: str, direction: str, apply_rules: bool = True, stale_after_s: int = 10
) -> dict[str, Any]:
    """The row holding the metric's extreme under the reading rules (apply_rules=False pools every row)."""
    cols = table["columns"]
    rows = [dict(zip(cols, r)) for r in table["rows"]]
    if apply_rules and "measured_at" in cols:
        snap = datetime.strptime(table["snapshot_time"], TIME_FMT)
        newest: dict[str, dict[str, Any]] = {}
        for r in rows:
            if r["cell_id"] not in newest or r["measured_at"] > newest[r["cell_id"]]["measured_at"]:
                newest[r["cell_id"]] = r
        for r in newest.values():
            if (snap - datetime.strptime(r["measured_at"], TIME_FMT)).total_seconds() > stale_after_s:
                raise ValueError(f"cell {r['cell_id']} has no row within {stale_after_s} s")
        rows = list(newest.values())
    if apply_rules and "source" in cols:
        e2_cells = {r["cell_id"] for r in rows if r["source"] == "E2_KPM"}
        rows = [r for r in rows if r["source"] == "E2_KPM" or r["cell_id"] not in e2_cells]
    best = max(r[metric] for r in rows) if direction == "max" else min(r[metric] for r in rows)
    hits = [r for r in rows if r[metric] == best]
    if len(hits) != 1:
        raise ValueError(f"{len(hits)} rows tie on the {direction} of {metric}")
    return hits[0]


def resolve_target_cluster(table: dict[str, Any], scope: str, apply_rules: bool = True, stale_after_s: int = 10) -> str:
    """target_cluster of a state-dependent scope under the truth rules; 'none' for any other scope."""
    if scope not in SCOPE_METRIC:
        return "none"
    metric, direction = SCOPE_METRIC[scope]
    return resolve_extreme(table, metric, direction, apply_rules, stale_after_s)["cluster"]


def count_tokens(text: str) -> int:
    """Token-length estimate (tiktoken cl100k_base); Laya truncation itself needs Laya's tokenizer."""
    return len(get_tokenizer().encode(text))
