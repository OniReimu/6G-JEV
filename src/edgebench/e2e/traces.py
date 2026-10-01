"""Trace generator and frozen trace manifest for Edgebench RQ5.

Generates reproducible arrival timelines, payload sizes, origin nodes, and request texts
for Part A (15 cells, 300 arrivals) and Part B (8 conditions, 240 arrivals).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Any

DEFAULT_POOL_PATH = Path("data/edgebench/v1/RQ2/clean/test.jsonl")

# 15 distinct cells per EXP-2026-002 design v0.2
CELLS_A: dict[str, dict[str, Any]] = {
    # Load sweep (lambda in {1, 2, 4, 8, 16}, N=5, D=2s, changing, cache off)
    "load_1": {"rate": 1.0, "deadline": 2.0, "nodes": 5, "text": "changing", "cache": False, "bursty": False},
    "load_2": {"rate": 2.0, "deadline": 2.0, "nodes": 5, "text": "changing", "cache": False, "bursty": False},
    "load_4": {"rate": 4.0, "deadline": 2.0, "nodes": 5, "text": "changing", "cache": False, "bursty": False},
    "load_8": {"rate": 8.0, "deadline": 2.0, "nodes": 5, "text": "changing", "cache": False, "bursty": False},
    "load_16": {"rate": 16.0, "deadline": 2.0, "nodes": 5, "text": "changing", "cache": False, "bursty": False},
    # Deadline sweep (D in {0.5, 1, 4}, lambda=4, N=5, changing, cache off)
    "deadline_0.5": {"rate": 4.0, "deadline": 0.5, "nodes": 5, "text": "changing", "cache": False, "bursty": False},
    "deadline_1": {"rate": 4.0, "deadline": 1.0, "nodes": 5, "text": "changing", "cache": False, "bursty": False},
    "deadline_4": {"rate": 4.0, "deadline": 4.0, "nodes": 5, "text": "changing", "cache": False, "bursty": False},
    # Topology sweep (N in {10, 20, 40}, lambda=8, D=2, changing, cache off)
    "topo_10": {"rate": 8.0, "deadline": 2.0, "nodes": 10, "text": "changing", "cache": False, "bursty": False},
    "topo_20": {"rate": 8.0, "deadline": 2.0, "nodes": 20, "text": "changing", "cache": False, "bursty": False},
    "topo_40": {"rate": 8.0, "deadline": 2.0, "nodes": 40, "text": "changing", "cache": False, "bursty": False},
    # Reuse block (lambda=4, N=5, D=2)
    "reuse_changing_on": {"rate": 4.0, "deadline": 2.0, "nodes": 5, "text": "changing", "cache": True, "bursty": False},
    "reuse_repeated_off": {"rate": 4.0, "deadline": 2.0, "nodes": 5, "text": "repeated", "cache": False, "bursty": False},
    "reuse_repeated_on": {"rate": 4.0, "deadline": 2.0, "nodes": 5, "text": "repeated", "cache": True, "bursty": False},
    # Bursty (alternating 0.5/8 in 20s segments, mean matched to 4, N=5, D=2, changing, cache off)
    "bursty": {"rate": 4.0, "deadline": 2.0, "nodes": 5, "text": "changing", "cache": False, "bursty": True},
}

# Aliases for flexible cell name parsing
CELL_ALIASES: dict[str, str] = {
    "load_1": "load_1", "load-1": "load_1", "lambda_1": "load_1", "lambda-1": "load_1", "lambda=1": "load_1", "load λ=1": "load_1",
    "load_2": "load_2", "load-2": "load_2", "lambda_2": "load_2", "lambda-2": "load_2", "lambda=2": "load_2", "load λ=2": "load_2",
    "load_4": "load_4", "load-4": "load_4", "lambda_4": "load_4", "lambda-4": "load_4", "lambda=4": "load_4", "load λ=4": "load_4", "centre": "load_4",
    "load_8": "load_8", "load-8": "load_8", "lambda_8": "load_8", "lambda-8": "load_8", "lambda=8": "load_8", "load λ=8": "load_8",
    "load_16": "load_16", "load-16": "load_16", "lambda_16": "load_16", "lambda-16": "load_16", "lambda=16": "load_16", "load λ=16": "load_16",
    "deadline_0.5": "deadline_0.5", "deadline-0.5": "deadline_0.5", "d_0.5": "deadline_0.5", "d=0.5": "deadline_0.5",
    "deadline_1": "deadline_1", "deadline-1": "deadline_1", "d_1": "deadline_1", "d=1": "deadline_1",
    "deadline_4": "deadline_4", "deadline-4": "deadline_4", "d_4": "deadline_4", "d=4": "deadline_4",
    "topo_10": "topo_10", "topo-10": "topo_10", "n_10": "topo_10", "n=10": "topo_10",
    "topo_20": "topo_20", "topo-20": "topo_20", "n_20": "topo_20", "n=20": "topo_20",
    "topo_40": "topo_40", "topo-40": "topo_40", "n_40": "topo_40", "n=40": "topo_40",
    "reuse_changing_on": "reuse_changing_on", "changing_on": "reuse_changing_on", "changing+on": "reuse_changing_on",
    "reuse_repeated_off": "reuse_repeated_off", "repeated_off": "reuse_repeated_off", "repeated+off": "reuse_repeated_off",
    "reuse_repeated_on": "reuse_repeated_on", "repeated_on": "reuse_repeated_on", "repeated+on": "reuse_repeated_on",
    "bursty": "bursty",
}


def normalize_cell_name(name: str) -> str:
    key = name.strip()
    if key in CELLS_A:
        return key
    if key in CELL_ALIASES:
        return CELL_ALIASES[key]
    low = key.lower().replace(" ", "")
    for k, target in CELL_ALIASES.items():
        if k.lower().replace(" ", "") == low:
            return target
    raise ValueError(f"Unknown cell name '{name}'. Available: {list(CELLS_A.keys())}")


def absolute_deadline(row: dict[str, Any], default_relative: float = 2.0) -> float:
    """Return absolute deadline: row['deadline'] if present, else row['arrival'] + default_relative."""
    if "deadline" in row and row["deadline"] is not None:
        return float(row["deadline"])
    return float(row["arrival"] + default_relative)


CONDITIONS_B: dict[str, dict[str, Any]] = {
    "steady_changing_off": {"rate": 2.0, "bursty": False, "text": "changing", "cache": False},
    "steady_changing_on": {"rate": 2.0, "bursty": False, "text": "changing", "cache": True},
    "steady_repeated_off": {"rate": 2.0, "bursty": False, "text": "repeated", "cache": False},
    "steady_repeated_on": {"rate": 2.0, "bursty": False, "text": "repeated", "cache": True},
    "bursty_changing_off": {"rate": 2.0, "bursty": True, "text": "changing", "cache": False},
    "bursty_changing_on": {"rate": 2.0, "bursty": True, "text": "changing", "cache": True},
    "bursty_repeated_off": {"rate": 2.0, "bursty": True, "text": "repeated", "cache": False},
    "bursty_repeated_on": {"rate": 2.0, "bursty": True, "text": "repeated", "cache": True},
}

ALIAS_MAP_B: dict[str, str] = {
    "steady-changing-cache-off": "steady_changing_off",
    "steady_changing_cache_off": "steady_changing_off",
    "steady-changing-off": "steady_changing_off",
    "steady-changing-cache0": "steady_changing_off",
    "steady-changing-cache-on": "steady_changing_on",
    "steady_changing_cache_on": "steady_changing_on",
    "steady-changing-on": "steady_changing_on",
    "steady-changing-cache1": "steady_changing_on",
    "steady-repeated-cache-off": "steady_repeated_off",
    "steady_repeated_cache_off": "steady_repeated_off",
    "steady-repeated-off": "steady_repeated_off",
    "steady-repeated-cache0": "steady_repeated_off",
    "steady-repeated-cache-on": "steady_repeated_on",
    "steady_repeated_cache_on": "steady_repeated_on",
    "steady-repeated-on": "steady_repeated_on",
    "steady-repeated-cache1": "steady_repeated_on",
    "bursty-changing-cache-off": "bursty_changing_off",
    "bursty_changing_cache_off": "bursty_changing_off",
    "bursty-changing-off": "bursty_changing_off",
    "bursty-changing-cache0": "bursty_changing_off",
    "bursty-changing-cache-on": "bursty_changing_on",
    "bursty_changing_cache_on": "bursty_changing_on",
    "bursty-changing-on": "bursty_changing_on",
    "bursty-changing-cache1": "bursty_changing_on",
    "bursty-repeated-cache-off": "bursty_repeated_off",
    "bursty_repeated_cache_off": "bursty_repeated_off",
    "bursty-repeated-off": "bursty_repeated_off",
    "bursty-repeated-cache0": "bursty_repeated_off",
    "bursty-repeated-cache-on": "bursty_repeated_on",
    "bursty_repeated_cache_on": "bursty_repeated_on",
    "bursty-repeated-on": "bursty_repeated_on",
    "bursty-repeated-cache1": "bursty_repeated_on",
}


def normalize_condition_b(name: str) -> str:
    key = name.strip()
    if key in CONDITIONS_B:
        return key
    if key in ALIAS_MAP_B:
        return ALIAS_MAP_B[key]
    low = key.lower().replace(" ", "_").replace("-", "_")
    for k in CONDITIONS_B:
        if k == low:
            return k
    raise ValueError(f"Unknown condition '{name}'. Available: {list(CONDITIONS_B.keys())}")


def load_ocr_selection(selection_path: Path | str = Path("data/edgebench/v1/ocr/selection.json")) -> dict[str, Any]:
    p = Path(selection_path)
    if not p.exists():
        raise FileNotFoundError(
            f"IIIT5K selection manifest not found at {p}. Run scripts/eb_iiit5k.py first."
        )
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_rq3_ocr_pool(
    paths: tuple[Path | str, ...] = (
        Path("data/edgebench/v1/RQ3/F4_low/test.jsonl"),
        Path("data/edgebench/v1/RQ3/F4_medium/test.jsonl"),
        Path("data/edgebench/v1/RQ3/F4_high/test.jsonl"),
    ),
    seed: int = 20260924,
    n_sample: int = 80,
) -> list[dict[str, Any]]:
    """Sample OCR texts from the union of RQ3 splits for Part B changing conditions."""
    seen_cases: dict[str, dict[str, Any]] = {}
    for p in paths:
        path_obj = Path(p)
        if not path_obj.exists():
            continue
        with open(path_obj, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                truth = row.get("truth", [])
                t0 = truth[0] if isinstance(truth, list) else truth
                if t0.get("service_type") == "ocr":
                    cid = row["case_id"]
                    if cid not in seen_cases:
                        seen_cases[cid] = row

    sorted_ocr = sorted(seen_cases.values(), key=lambda r: r["case_id"])
    rng = random.Random(seed)
    return rng.sample(sorted_ocr, n_sample)



def load_pool(path: str | Path = DEFAULT_POOL_PATH) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Pool file not found: {p}")
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def select_repeated_8(pool: list[dict[str, Any]], seed: int = 20260924) -> list[dict[str, Any]]:
    """Select 8 fixed texts (6 supported OCR and 2 non-OCR/unsupported) deterministically."""
    rng = random.Random(seed)
    
    # Check if pool explicitly has 'unsupported' service_type
    unsupported_candidates = [
        r for r in pool
        if (r.get("truth") and r["truth"][0].get("service_type") == "unsupported")
        or r.get("meta", {}).get("is_unsupported")
    ]
    
    ocr_candidates = [
        r for r in pool
        if r.get("truth") and r["truth"][0].get("service_type") == "ocr"
    ]
    
    sorted_ocr = sorted(ocr_candidates, key=lambda r: r["case_id"])
    selected_ocr = rng.sample(sorted_ocr, 6)
    
    if len(unsupported_candidates) >= 2:
        sorted_unsupp = sorted(unsupported_candidates, key=lambda r: r["case_id"])
        selected_other = rng.sample(sorted_unsupp, 2)
    else:
        # From clean/test.jsonl: select 1 count and 1 detection (non-OCR services)
        counts = sorted(
            [r for r in pool if r.get("truth") and r["truth"][0].get("service_type") == "count"],
            key=lambda r: r["case_id"],
        )
        dets = sorted(
            [r for r in pool if r.get("truth") and r["truth"][0].get("service_type") == "detection"],
            key=lambda r: r["case_id"],
        )
        selected_other = [rng.choice(counts), rng.choice(dets)]
    
    result = selected_ocr + selected_other
    assert len(result) == 8
    return result


def generate_trace_a(
    cell: str,
    seed: int = 1,
    pool: list[dict[str, Any]] | None = None,
    n: int = 300,
    pool_path: str | Path = DEFAULT_POOL_PATH,
) -> list[dict[str, Any]]:
    """Generate deterministic arrival trace for Part A (N-node modeled execution)."""
    norm_cell = normalize_cell_name(cell)
    cfg = CELLS_A[norm_cell]
    
    if pool is None:
        pool = load_pool(pool_path)
    
    deadline = cfg["deadline"]
    nodes = cfg["nodes"]
    is_bursty = cfg["bursty"]
    rate = cfg["rate"]
    text_mode = cfg["text"]
    
    repeated_8 = select_repeated_8(pool, seed=20260924)
    
    rng = random.Random(seed)
    
    # 1. Generate text assignment
    if text_mode == "changing":
        if len(pool) < n:
            raise ValueError(f"Pool size {len(pool)} is smaller than arrivals {n}")
        pool_copy = list(pool)
        rng.shuffle(pool_copy)
        chosen_items = pool_copy[:n]
    else:
        # Repeated: sample from the 8 fixed texts
        chosen_items = [repeated_8[rng.randrange(len(repeated_8))] for _ in range(n)]
    
    # 2. Generate arrival times, origins, payload sizes
    arrivals = []
    t = 0.0
    for i in range(n):
        if not is_bursty:
            t += rng.expovariate(rate)
        else:
            # Alternating 0.5 and 8 req/s in 20s segments
            while True:
                segment = int(t // 20)
                cur_rate = 0.5 if segment % 2 == 0 else 8.0
                proposal = t + rng.expovariate(cur_rate)
                if proposal >= (segment + 1) * 20:
                    t = float((segment + 1) * 20)
                    continue
                t = proposal
                break
        
        origin = rng.randrange(nodes)
        size_mb = rng.uniform(0.25, 2.0)
        item = chosen_items[i]
        
        truth = item["truth"]
        truth_dict = truth[0] if isinstance(truth, list) else truth
        
        arrivals.append({
            "id": i,
            "arrival": round(t, 6),
            "deadline": round(t + deadline, 6),
            "origin": origin,
            "size_mb": round(size_mb, 4),
            "case_id": item["case_id"],
            "text": item["text"],
            "truth": [dict(truth_dict)] if isinstance(truth, list) else dict(truth_dict),
            "fields": item.get("fields", ["service_type", "locality", "quality_floor", "urgency"]),
            "meta": dict(item.get("meta", {})),
            "cell": norm_cell,
            "seed": seed,
            "cache_enabled": cfg["cache"],
        })
    
    return arrivals


def generate_trace_b(
    condition: str,
    seed: int = 1,
    selection: dict[str, Any] | None = None,
    pool: list[dict[str, Any]] | None = None,
    n_ocr: int = 180,
    n_other: int = 60,
    deadline: float = 2.0,
) -> list[dict[str, Any]]:
    """Generate 240 arrivals for Part B: 180 OCR requests paired with test images, 60 non-OCR requests."""
    norm_cond = normalize_condition_b(condition)
    cfg = CONDITIONS_B[norm_cond]

    if selection is None:
        selection = load_ocr_selection()
    if pool is None:
        pool = load_pool()

    test_images = selection["test_images"]  # 200 images
    if len(test_images) < n_ocr:
        raise ValueError(f"Need at least {n_ocr} test images, got {len(test_images)}")

    ocr_texts = [r for r in pool if r.get("truth") and (r["truth"][0] if isinstance(r["truth"], list) else r["truth"]).get("service_type") == "ocr"]
    non_ocr_texts = [r for r in pool if r.get("truth") and (r["truth"][0] if isinstance(r["truth"], list) else r["truth"]).get("service_type") != "ocr"]

    repeated_8 = select_repeated_8(pool, seed=20260924)
    repeated_ocr = [r for r in repeated_8 if (r["truth"][0] if isinstance(r["truth"], list) else r["truth"]).get("service_type") == "ocr"]
    repeated_other = [r for r in repeated_8 if (r["truth"][0] if isinstance(r["truth"], list) else r["truth"]).get("service_type") != "ocr"]

    rng = random.Random(seed)
    is_bursty = cfg["bursty"]
    text_mode = cfg["text"]
    rate = cfg["rate"]
    total_arrivals = n_ocr + n_other  # 240

    # Assign texts and images
    if text_mode == "changing":
        # 100 clean OCR texts + 80 sampled from RQ3 splits
        sampled_80 = load_rq3_ocr_pool()
        all_changing_ocr = ocr_texts + sampled_80
        chosen_ocr_texts = all_changing_ocr[:n_ocr]
        # Sample 60 non-OCR texts
        chosen_other_texts = rng.sample(non_ocr_texts, n_other)
    else:
        # Repeated: 30 repetitions of the 6 OCR texts and 2 non-OCR texts
        chosen_ocr_texts = [repeated_ocr[i % len(repeated_ocr)] for i in range(n_ocr)]
        chosen_other_texts = [repeated_other[i % len(repeated_other)] for i in range(n_other)]

    # Pair the 180 OCR texts with 180 test images
    ocr_items = []
    for i in range(n_ocr):
        text_row = chosen_ocr_texts[i]
        img = test_images[i]
        ocr_items.append({
            "is_ocr": True,
            "case_id": text_row["case_id"],
            "text": text_row["text"],
            "truth": text_row["truth"],
            "fields": text_row.get("fields", ["service_type", "locality", "quality_floor", "urgency"]),
            "image_id": img["id"],
            "image_path": img["path"],
            "image_sha256": img["sha256"],
            "ground_truth": img["ground_truth"],
        })

    other_items = []
    dummy_img = test_images[0]
    for i in range(n_other):
        text_row = chosen_other_texts[i]
        other_items.append({
            "is_ocr": False,
            "case_id": text_row["case_id"],
            "text": text_row["text"],
            "truth": text_row["truth"],
            "fields": text_row.get("fields", ["service_type", "locality", "quality_floor", "urgency"]),
            "image_id": dummy_img["id"],
            "image_path": dummy_img["path"],
            "image_sha256": dummy_img["sha256"],
            "ground_truth": "",
        })

    # Interleave OCR and non-OCR items with 3:1 ratio deterministically
    combined_items = []
    idx_ocr, idx_other = 0, 0
    # Create pattern of 3 OCR, 1 other
    while idx_ocr < n_ocr or idx_other < n_other:
        for _ in range(3):
            if idx_ocr < n_ocr:
                combined_items.append(ocr_items[idx_ocr])
                idx_ocr += 1
        if idx_other < n_other:
            combined_items.append(other_items[idx_other])
            idx_other += 1

    # Generate arrival times
    arrivals = []
    t = 0.0
    for i in range(total_arrivals):
        if not is_bursty:
            t += rng.expovariate(rate)
        else:
            while True:
                segment = int(t // 20)
                cur_rate = 0.5 if segment % 2 == 0 else 8.0
                proposal = t + rng.expovariate(cur_rate)
                if proposal >= (segment + 1) * 20:
                    t = float((segment + 1) * 20)
                    continue
                t = proposal
                break

        item = combined_items[i]
        arrivals.append({
            "id": i,
            "arrival": round(t, 6),
            "deadline": round(t + deadline, 6),
            "is_ocr": item["is_ocr"],
            "case_id": item["case_id"],
            "text": item["text"],
            "truth": item["truth"],
            "fields": item["fields"],
            "image_id": item["image_id"],
            "image_path": item["image_path"],
            "image_sha256": item["image_sha256"],
            "ground_truth": item["ground_truth"],
            "condition": norm_cond,
            "seed": seed,
            "cache_enabled": cfg["cache"],
        })

    return arrivals



def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def generate_all_traces_a(
    out_dir: str | Path = "traces",
    seeds: tuple[int, ...] = (1,),
    pool_path: str | Path = DEFAULT_POOL_PATH,
) -> dict[str, Any]:
    """Generate and freeze all Part A traces per (cell, seed) with SHA-256 manifest."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pool = load_pool(pool_path)
    
    manifest_records: dict[str, dict[str, Any]] = {}
    
    for cell in CELLS_A:
        cell_dir = out / cell
        cell_dir.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            trace = generate_trace_a(cell, seed=seed, pool=pool)
            trace_path = cell_dir / f"{seed}.jsonl"
            with open(trace_path, "w", encoding="utf-8") as f:
                for row in trace:
                    f.write(json.dumps(row) + "\n")
            
            sha = compute_sha256(trace_path)
            rel_path = str(trace_path.relative_to(out))
            manifest_records[f"{cell}/{seed}"] = {
                "file": rel_path,
                "cell": cell,
                "seed": seed,
                "arrivals": len(trace),
                "sha256": sha,
            }
    
    manifest_path = out / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_records, f, indent=2)
    
    return manifest_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and freeze Edgebench Part A traces.")
    parser.add_argument("--out", default="traces", help="Output directory for traces")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1], help="Seeds to generate")
    parser.add_argument("--pool", default=str(DEFAULT_POOL_PATH), help="Path to text pool")
    args = parser.parse_args()
    
    records = generate_all_traces_a(out_dir=args.out, seeds=tuple(args.seeds), pool_path=args.pool)
    print(f"Generated {len(records)} trace files under {args.out}/ (manifest: {args.out}/manifest.json)")


if __name__ == "__main__":
    main()
