"""Assembly of corpus files, timing subsets, manifest.json with SHA-256, and stats.md."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import tiktoken

from src.edgebench.contract import Case, dump_cases_jsonl, load_cases_jsonl
from src.edgebench.corpus.pad import get_tokenizer
from src.edgebench.ledger import get_git_status

TIMING_SEED = 20260924
FROZEN_FILE = "_frozen.json"


def compute_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_frozen(data_dir: Path | str) -> dict[str, dict[str, Any]]:
    """Frozen groups: {"<RQ>/<condition>/<split>": {sha256, n, frozen_at, git_head}}."""
    path = Path(data_dir) / FROZEN_FILE
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def frozen_group_key(data_dir: Path | str, path: Path | str) -> str:
    """Group key of a formal file: its path relative to data_dir without .jsonl (e.g. RQ2/clean/test)."""
    return Path(path).relative_to(Path(data_dir)).with_suffix("").as_posix()


def check_frozen_outputs(data_dir: Path | str, outputs: dict[Path, list[Case]]) -> dict[Path, bytes]:
    """Serialize outputs exactly as dump_cases_jsonl writes them.

    Raises (before anything is written) if an output belongs to a frozen group and is not byte-identical
    to the frozen file.
    """
    frozen = load_frozen(data_dir)
    blobs: dict[Path, bytes] = {}
    changed: list[str] = []
    for path, cases in outputs.items():
        data = "".join(json.dumps(c.to_dict()) + "\n" for c in cases).encode("utf-8")
        key = frozen_group_key(data_dir, path)
        if key in frozen and hashlib.sha256(data).hexdigest() != frozen[key]["sha256"]:
            changed.append(key)
        blobs[path] = data
    if changed:
        raise ValueError(f"frozen group(s) would change: {sorted(changed)}; nothing written")
    return blobs


def write_frozen_outputs(data_dir: Path | str, blobs: dict[Path, bytes], freeze: bool) -> list[Path]:
    """Write checked blobs; with freeze, record each not-yet-frozen group in _frozen.json."""
    for path, data in blobs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    if freeze:
        frozen = load_frozen(data_dir)
        git_head, _ = get_git_status()
        frozen_at = datetime.now(timezone.utc).isoformat()
        for path, data in blobs.items():
            # A frozen group already passed check_frozen_outputs byte-identical: keep its original record
            frozen.setdefault(frozen_group_key(data_dir, path), {
                "sha256": hashlib.sha256(data).hexdigest(),
                "n": data.count(b"\n"),
                "frozen_at": frozen_at,
                "git_head": git_head,
            })
        frozen_file = Path(data_dir) / FROZEN_FILE
        tmp_file = frozen_file.with_suffix(".tmp")
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(dict(sorted(frozen.items())), f, indent=2)
        tmp_file.replace(frozen_file)
    return list(blobs)


def select_stratified_timing_subset(
    test_cases: list[Case], n_subset: int = 100, seed: int = TIMING_SEED
) -> list[str]:
    """Select n_subset case IDs stratified by (wording_family, service)."""
    if len(test_cases) <= n_subset:
        return [c.case_id for c in test_cases]

    # Group by stratum
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for c in test_cases:
        wording = str(c.meta.get("wording_family", "unknown"))
        svc = str(c.meta.get("target_service", c.truth[0].get("service_type", "unknown")))
        strata[(wording, svc)].append(c.case_id)

    rng = random.Random(seed)
    # Shuffle each stratum
    for key in strata:
        rng.shuffle(strata[key])

    selected: list[str] = []
    strata_keys = sorted(strata.keys())
    rng.shuffle(strata_keys)

    # Round-robin selection across strata
    while len(selected) < n_subset:
        added_in_round = False
        for k in strata_keys:
            if strata[k]:
                selected.append(strata[k].pop(0))
                added_in_round = True
                if len(selected) == n_subset:
                    break
        if not added_in_round:
            break

    selected.sort()
    return selected


def write_timing_subset_file(dest_path: Path | str, case_ids: list[str]) -> Path:
    target = Path(dest_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        for cid in case_ids:
            f.write(f"{cid}\n")
    return target


def generate_manifest(
    root_dir: Path | str,
    yields: dict[str, dict[str, Any]],
    total_cost_usd: float,
    seed: int,
    prompt_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    root = Path(root_dir)
    file_hashes: dict[str, str] = {}

    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            rel_path = str(path.relative_to(root))
            file_hashes[rel_path] = compute_sha256(path)

    catalog_file_hashes = {
        str(path.relative_to(root)): compute_sha256(path)
        for path in sorted((root / "catalog").glob("*.json"))
    }

    # Per-item provenance exactly as recorded on the case: derived items carry their parent's strings
    # (a bundle carries the list of its parents'); nothing falls back to a default model name.
    items_meta: dict[str, dict[str, Any]] = {}
    generators_set: set[str] = set()
    verifiers_set: set[str] = set()

    for path in sorted(root.glob("*/*/*.jsonl")):
        try:
            cases = load_cases_jsonl(path)
            for c in cases:
                gen = c.meta.get("generator") or c.meta.get("generator_model") or None
                ver = c.meta.get("verifier") or c.meta.get("verifier_model") or None
                items_meta[c.case_id] = {
                    "generator": gen,
                    "verifier": ver,
                    "prompt_sha256": c.meta.get("prompt_sha256"),
                    "derivation": c.meta.get("derivation"),
                }
                for g in gen if isinstance(gen, list) else [gen]:
                    if g:
                        generators_set.add(str(g))
                for v in ver if isinstance(ver, list) else [ver]:
                    if v:
                        verifiers_set.add(str(v))
        except Exception:
            pass

    generators_list = sorted(generators_set)
    verifiers_list = sorted(verifiers_set)
    git_head, tree_dirty = get_git_status()

    manifest_data = {
        "seed": seed,
        "git_head": git_head,
        "tree_dirty": tree_dirty,
        "catalog_file_hashes": catalog_file_hashes,
        "models": {
            "generator": generators_list[0] if len(generators_list) == 1 else None,
            "verifier": verifiers_list[0] if len(verifiers_list) == 1 else None,
            "generators": generators_list,
            "verifiers": verifiers_list,
        },
        "generators": generators_list,
        "verifiers": verifiers_list,
        "items": items_meta,
        "yields": yields,
        "total_cost_usd": total_cost_usd,
        "prompt_hashes": prompt_hashes or {},
        "files": file_hashes,
    }

    manifest_file = root / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    return manifest_data


def generate_stats_markdown(root_dir: Path | str, yields: dict[str, dict[str, Any]]) -> str:
    root = Path(root_dir)
    enc = get_tokenizer()

    lines = [
        "# EdgeIntent v1 Corpus Statistics\n",
        "| Condition | Split | N | Length p5 | Length p50 | Length p95 | First-pass yield | Final yield |",
        "|---|---|---|---|---|---|---|---|",
    ]

    field_marginals_md = ["\n## Field Marginals per Condition\n"]

    for cond_dir in sorted(root.glob("*/*")):
        if not cond_dir.is_dir():
            continue
        rq = cond_dir.parent.name
        cond = cond_dir.name
        cond_key = f"{rq}/{cond}"

        for split in ("test", "dev"):
            split_file = cond_dir / f"{split}.jsonl"
            if not split_file.exists():
                continue

            cases = load_cases_jsonl(split_file)
            n_cases = len(cases)
            if n_cases == 0:
                continue

            lengths = [len(enc.encode(c.text)) for c in cases]
            p5 = float(np.percentile(lengths, 5))
            p50 = float(np.percentile(lengths, 50))
            p95 = float(np.percentile(lengths, 95))
            # Yields come from _yields.json (generated conditions only); derived conditions have none.
            y = yields.get(cond)
            fp_str = f"{y['first_pass']:.1%}" if y else "n/a"
            final_str = f"{y['final']:.1%}" if y else "n/a"

            lines.append(
                f"| {cond_key} | {split} | {n_cases} | {p5:.1f} | {p50:.1f} | {p95:.1f} | {fp_str} | {final_str} |"
            )

            # Record field marginals
            field_marginals_md.append(f"### {cond_key} ({split}, n={n_cases})\n")
            fields = cases[0].fields
            for f in fields:
                vals = [t.get(f, "missing") for c in cases for t in c.truth]
                counts = Counter(vals)
                counts_str = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
                field_marginals_md.append(f"- **{f}**: {counts_str}")
            field_marginals_md.append("")

    full_md = "\n".join(lines) + "\n" + "\n".join(field_marginals_md)
    stats_file = root / "stats.md"
    with open(stats_file, "w", encoding="utf-8") as f:
        f.write(full_md)

    return full_md
