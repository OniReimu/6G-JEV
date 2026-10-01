"""Assembly of the frozen RANIntent v1 corpus with a SHA-256 manifest.

Layout under data_dir:
  config.json, schemas/*                      - the config and shared schema/rules files used
  tuples/<split>.jsonl                        - one record per tuple: truth, issuer, text, provenance
  RQ4/c<n>_<quality>/<split>.jsonl            - 16 conditions: the same text with that condition's telemetry
  pool/c2c3_pool.jsonl                        - named-scope half of (21 cells, fresh), test split, with C2 class
  stats.json                                  - token estimates (Laya 1,024-token context) and condition shares
  manifest.json                               - sha256 of every file above, seed, models, counts, yields
  provenance.json                             - assembly time and git state (outside the identity check)
Once a manifest exists the corpus is frozen: a later assembly must reproduce every file and manifest.json byte for
byte, or nothing is written.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.edgebench.corpus.tuples import TupleItem
from src.edgebench.ledger import get_git_status
from src.ranbench.corpus.exchange import RanExchange, to_tuple_item
from src.ranbench.corpus.spec import (
    ACTUATED_FIELDS,
    FIELDS,
    build_policy_type,
    build_reading_rules,
    build_truth_schema,
    conditions,
    issuer_block,
    validate_truth,
)
from src.ranbench.corpus.stub import STUB_GENERATOR, STUB_VERIFIER
from src.ranbench.corpus.telemetry import build_table, count_tokens, render_table, resolve_target_cluster
from src.ranbench.corpus.tuples import sample_tuples, truth_record
from src.ranbench.schemas import POLICY_TYPE_PATH, READING_RULES_PATH, TRUTH_SCHEMA_PATH


def schema_file_contents(cfg: dict[str, Any]) -> dict[Path, str]:
    """Expected contents of the shared schema/rules files, generated from spec."""
    return {
        POLICY_TYPE_PATH: json.dumps(build_policy_type(cfg), indent=2) + "\n",
        TRUTH_SCHEMA_PATH: json.dumps(build_truth_schema(cfg), indent=2) + "\n",
        READING_RULES_PATH: build_reading_rules(cfg),
    }


def write_schema_files(cfg: dict[str, Any]) -> list[Path]:
    for path, text in schema_file_contents(cfg).items():
        path.write_text(text, encoding="utf-8")
    return list(schema_file_contents(cfg))


def case_record(
    item: TupleItem, text: str, n_cells: int, quality: str, cfg: dict[str, Any], provenance: dict[str, Any]
) -> dict[str, Any]:
    """One RQ4 case. For a state-dependent tuple, the reference reader must reproduce target_cluster."""
    table, info = build_table(item, n_cells, quality, cfg)
    telemetry = render_table(table)
    lab = item.tuple_labels
    meta: dict[str, Any] = {
        **provenance,
        "telemetry_sha256": hashlib.sha256(telemetry.encode("utf-8")).hexdigest(),
        "tokens_cl100k": {"telemetry": count_tokens(telemetry), "text": count_tokens(text)},
        "extreme_cells": info["extreme_cells"],
        "decoy_cell": info["decoy_cell"],
    }
    if item.meta["half"] == "state_dependent":
        stale_after = cfg["telemetry"]["stale_after_s"]
        got = resolve_target_cluster(table, lab["scope"], True, stale_after)
        if got != lab["target_cluster"]:
            raise ValueError(f"{item.tuple_id} c{n_cells}_{quality}: reader gives {got}, truth {lab['target_cluster']}")
        meta["rules_ignored_cluster"] = resolve_target_cluster(table, lab["scope"], False, stale_after)
    return {
        "case_id": item.tuple_id,
        "condition": f"c{n_cells}_{quality}",
        "n_cells": n_cells,
        "quality": quality,
        "split": item.split,
        "half": item.meta["half"],
        "issuer": issuer_block(item.meta["issuer"], cfg),
        "text": text,
        "telemetry": telemetry,
        "truth": {f: lab[f] for f in FIELDS},
        "meta": meta,
    }


def _jsonl(records: list[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(r) + "\n" for r in records).encode("utf-8")


def _stats(cases_by_cond: dict[str, dict[str, list[dict[str, Any]]]], cfg: dict[str, Any]) -> dict[str, Any]:
    limit = cfg["tokens"]["laya_context_tokens"]
    out: dict[str, Any] = {"token_encoding": cfg["tokens"]["estimate_encoding"], "laya_context_tokens": limit, "conditions": {}}
    for cond, by_split in cases_by_cond.items():
        out["conditions"][cond] = {}
        for sp, cases in by_split.items():
            tok = np.array([c["meta"]["tokens_cl100k"]["telemetry"] for c in cases])
            sd = [c for c in cases if c["half"] == "state_dependent"]
            out["conditions"][cond][sp] = {
                "n": len(cases),
                "telemetry_tokens": {"p50": float(np.percentile(tok, 50)), "p95": float(np.percentile(tok, 95)), "max": int(tok.max())},
                "share_telemetry_over_laya_context": float((tok > limit).mean()),
                "state_dependent_n": len(sd),
                "state_dependent_decoy_share": sum(c["meta"]["decoy_cell"] is not None for c in sd) / len(sd) if sd else 0.0,
                "state_dependent_rules_ignored_wrong_share": (
                    sum(c["meta"]["rules_ignored_cluster"] != c["truth"]["target_cluster"] for c in sd) / len(sd) if sd else 0.0
                ),
            }
    return out


def assemble_corpus(exchange: RanExchange, allow_stub: bool = False) -> dict[str, Any]:
    """Build every corpus file from the accepted items and freeze them under manifest.json."""
    cfg = exchange.cfg
    data_dir = Path(exchange.data_dir)
    stale = [p.name for p, text in schema_file_contents(cfg).items() if not p.exists() or p.read_text(encoding="utf-8") != text]
    if stale:
        raise ValueError(f"schema files out of date with spec: {stale}; run write-schemas")

    tuples: dict[str, list[TupleItem]] = {}
    texts: dict[str, dict[str, Any]] = {}
    for sp in cfg["splits"]:
        tuples[sp] = sample_tuples(sp, cfg)
        accepted = [it for it in exchange.items.values() if it.split == sp and it.status == "accepted"]
        by_tuple: dict[str, list[Any]] = {}
        for it in accepted:
            by_tuple.setdefault(it.tuple_id, []).append(it)
        missing = [t.tuple_id for t in tuples[sp] if len(by_tuple.get(t.tuple_id, [])) != 1]
        if missing or len(by_tuple) != len(tuples[sp]):
            raise ValueError(f"assemble {sp}: {len(missing)} tuple(s) without exactly one accepted item, e.g. {missing[:5]}")
        for it in accepted:
            if not allow_stub and (it.generator == STUB_GENERATOR or it.verifier == STUB_VERIFIER):
                raise ValueError(f"assemble {sp}: stub-generated or stub-verified item {it.item_id}; pass allow_stub for a dry run")
            if to_tuple_item(it).tuple_labels != next(t for t in tuples[sp] if t.tuple_id == it.tuple_id).tuple_labels:
                raise ValueError(f"assemble {sp}: state truth of {it.tuple_id} differs from the sampler")
            texts[it.tuple_id] = {
                "text": it.text,
                "provenance": {
                    "wording_family": it.meta["wording_family"],
                    "generator": it.generator,
                    "verifier": it.verifier,
                    "attempts": it.attempt,
                    "is_replacement": it.is_replacement,
                    "prompt_sha256": it.meta.get("prompt_sha256"),
                },
            }

    outputs: dict[str, bytes] = {"config.json": (json.dumps(cfg, indent=2) + "\n").encode("utf-8")}
    for path, text in schema_file_contents(cfg).items():
        outputs[f"schemas/{path.name}"] = text.encode("utf-8")
    cases_by_cond: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for sp, items in tuples.items():
        records = []
        for t in items:
            truth = truth_record(t)
            problems = validate_truth(truth, cfg)
            if problems:
                raise ValueError(f"{t.tuple_id}: invalid truth: {problems}")
            records.append({
                "tuple_id": t.tuple_id, "split": sp, "half": t.meta["half"], "truth": truth,
                "issuer": issuer_block(t.meta["issuer"], cfg), "text": texts[t.tuple_id]["text"],
                "meta": {"item_index": t.meta["item_index"], **texts[t.tuple_id]["provenance"]},
            })
        outputs[f"tuples/{sp}.jsonl"] = _jsonl(records)
        for cond, n, q in conditions(cfg):
            cases = [case_record(t, texts[t.tuple_id]["text"], n, q, cfg, texts[t.tuple_id]["provenance"]) for t in items]
            cases_by_cond.setdefault(cond, {})[sp] = cases
            outputs[f"RQ4/{cond}/{sp}.jsonl"] = _jsonl(cases)

    pc = cfg["pool"]
    pool = []
    for c in cases_by_cond[f"c{pc['cells']}_{pc['quality']}"][pc["split"]]:
        if c["half"] != pc["half"]:
            continue
        pool.append({
            "intent_id": c["case_id"], "source_condition": c["condition"], "split": c["split"],
            "issuer": c["issuer"], "text": c["text"],
            "actuated": {f: c["truth"][f] for f in ACTUATED_FIELDS},
            "c2_class": cfg["c2_class_map"][c["truth"]["class"]],
            "truth": c["truth"],
        })
    outputs["pool/c2c3_pool.jsonl"] = _jsonl(pool)
    outputs["stats.json"] = (json.dumps(_stats(cases_by_cond, cfg), indent=2) + "\n").encode("utf-8")

    hashes = {rel: hashlib.sha256(data).hexdigest() for rel, data in sorted(outputs.items())}
    frozen = exchange.frozen_files()
    if frozen and frozen != hashes:
        changed = sorted(k for k in set(frozen) | set(hashes) if frozen.get(k) != hashes.get(k))
        raise ValueError(f"corpus is frozen and {len(changed)} file(s) would change: {changed[:10]}; nothing written")

    accepted_all = [it for it in exchange.items.values() if it.status == "accepted"]
    manifest = {
        "name": cfg["name"],
        "experiment": cfg["experiment"],
        "seed": cfg["base_seed"],
        "stub": any(it.generator == STUB_GENERATOR or it.verifier == STUB_VERIFIER for it in accepted_all),
        "generators": sorted({it.generator for it in accepted_all if it.generator}),
        "verifiers": sorted({it.verifier for it in accepted_all if it.verifier}),
        "counts": {
            "tuples": {sp: len(v) for sp, v in tuples.items()},
            "conditions": len(cases_by_cond),
            "pool": len(pool),
        },
        "yields": exchange.yields(),
        "files": hashes,
    }
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    manifest_path = data_dir / "manifest.json"
    if frozen and manifest_path.read_bytes() != manifest_bytes:
        raise ValueError("corpus is frozen and manifest.json would change; nothing written")
    for rel, data in outputs.items():
        path = data_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    if frozen:
        return manifest
    # Wall clock and git state live outside the identity-checked manifest, written once at the freeze
    git_head, tree_dirty = get_git_status()
    provenance = {
        "assembled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_head": git_head,
        "tree_dirty": tree_dirty,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }
    (data_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    manifest_path.write_bytes(manifest_bytes)
    return manifest
