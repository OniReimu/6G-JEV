"""File-exchange transport for EdgeIntent v1 corpus generation and verification.

Directory protocol:
  runs/_corpus_exchange/
    gen/todo/<batch_id>.jsonl     - exported by pipeline
    gen/done/<batch_id>.jsonl     - written by external generator
    ver/todo/<batch_id>.jsonl     - exported for blind verification (item_id, text ONLY)
    ver/todo/<batch_id>.context.md - field criteria & instructions (no target info)
    ver/done/<batch_id>.jsonl     - written by external verifier (item_id, labels, verifier)
    _targets/                     - target tuples and state index (never exposed to verifier)
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any

from src.edgebench.contract import Case, DEFAULT_CRITERIA, load_cases_jsonl
from src.edgebench.corpus.assemble import check_frozen_outputs, load_frozen, write_frozen_outputs
from src.edgebench.corpus.catalog import (
    build_nested_and_churn_catalogs,
    get_services_catalog,
    load_catalogs,
    load_services,
    save_catalogs,
)
from src.edgebench.corpus.lints import (
    check_catalog_leak_lint,
    check_codeswitch_monolingual_lint,
    check_copy_4gram_lint,
    check_diversity_lint,
    check_enum_leak_lint,
    check_scaffold_lint,
    extract_ngrams,
    normalize_words,
)
from src.edgebench.corpus.prompts import (
    build_single_item_generator_prompt,
    build_verifier_context_markdown,
)
from src.edgebench.corpus.tuples import (
    TupleItem,
    is_rq4_condition,
    rq4_active_catalog_and_unsupported_pool,
    sample_tuples_for_condition,
)

ALL_CONDITIONS: list[str] = [
    # RQ2
    "clean",
    "colloquial",
    "negation",
    "codeswitch",
    "defaultbait",
    "revised",
    "keyvalue",
    # RQ3
    "F4_low",
    "F4_medium",
    "F4_high",
    "F6_low",
    "F6_medium",
    "F6_high",
    "F8_low",
    "F8_medium",
    "F8_high",
    # RQ4
    "K4",
    "K15",
    "K64",
    "K128",
    "K254",
    "churn25",
    "churn50",
]


def get_rq_for_condition(condition: str) -> str:
    norm_cond = condition.split("/")[-1]
    if norm_cond in ("clean", "colloquial", "negation", "codeswitch", "defaultbait", "revised", "keyvalue", "noise"):
        return "RQ2"
    if norm_cond.startswith("F4_") or norm_cond.startswith("F6_") or norm_cond.startswith("F8_"):
        return "RQ3"
    if is_rq4_condition(norm_cond):
        return "RQ4"
    if norm_cond.startswith("pad_") or norm_cond == "base":
        return "RQ1a"
    if norm_cond.startswith("k") and len(norm_cond) <= 3:
        return "RQ1b"
    return "RQ2"


def group_key(condition: str, split: str) -> str:
    """Key of a (condition, split) group in _frozen.json: its formal file path under data_dir, minus .jsonl."""
    return f"{get_rq_for_condition(condition)}/{condition}/{split}"


def _check_done_matches_todo(stage: str, done_file: Path, todo_file: Path, done_ids: set[str]) -> None:
    """A done file must carry exactly the item IDs of its todo batch; otherwise it is not ingested."""
    if not todo_file.exists():
        raise ValueError(f"Malformed {stage} done file {done_file.name}: no matching todo batch {todo_file.name}")
    todo_ids: set[str] = set()
    with open(todo_file, "r", encoding="utf-8") as f:
        for l in f:
            if l.strip():
                todo_ids.add(json.loads(l)["item_id"])
    extra = done_ids - todo_ids
    if extra:
        raise ValueError(f"Malformed {stage} done file {done_file.name}: contains extra item IDs not in todo batch: {sorted(extra)}")
    missing = todo_ids - done_ids
    if missing:
        raise ValueError(
            f"Malformed {stage} done file {done_file.name}: item IDs do not match todo batch; "
            f"missing {len(missing)} of {len(todo_ids)}: {sorted(missing)}"
        )


def make_opaque_item_id(
    tuple_id: str,
    condition: str,
    split: str,
    attempt: int = 1,
    rep_index: int = 0,
) -> str:
    """Generate an opaque item ID that contains no condition name or target value."""
    key = f"{condition}:{split}:{tuple_id}:rep{rep_index}"
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return f"it_{h}"


def tuple_group_key(it: TargetItem) -> tuple[Any, ...]:
    """Items that render the same label tuple in parallel texts share this key.

    tuple_id is "<condition>_<split>_<index:04d>" (tuples.sample_tuples_for_condition), and the index
    selects the tuple from a seed shared across conditions:
      RQ2 - all seven conditions share one tuple set per split (seed key "RQ2_shared", D-6)
            -> ("RQ2", split, index)
      RQ3 - F*_low/medium/high at one F share tuple, scenario and wording family (seed key "F<n>_shared")
            -> ("RQ3", "F<n>", split, index)
      RQ4 - no sharing across conditions -> ("RQ4", item_id)
    Blind verification must never put two items with one key in the same batch.
    """
    index = int(it.tuple_id.rsplit("_", 1)[1])
    rq = get_rq_for_condition(it.condition)
    if rq == "RQ2":
        return ("RQ2", it.split, index)
    if rq == "RQ3":
        return ("RQ3", it.condition.split("/")[-1].split("_")[0], it.split, index)
    return (rq, it.item_id)


@dataclass
class TargetItem:
    item_id: str
    tuple_id: str
    condition: str
    split: str
    tuple_labels: dict[str, str]
    meta: dict[str, Any]
    attempt: int = 1
    is_replacement: bool = False
    replaced_from: str | None = None
    rep_index: int = 0
    status: str = "pending_gen"  # pending_gen, exported_gen, done_gen, exported_ver, accepted, rejected
    gen_batch_id: str | None = None
    ver_batch_id: str | None = None
    text: str | None = None
    generator: str | None = None
    verifier_labels: dict[str, str] | None = None
    verifier: str | None = None
    reject_reasons: list[str] = field(default_factory=list)
    # requeue/retarget history: requeue_ver {requeued_from, reason, archive}; requeue_gen and retarget add "op" and "time"
    requeues: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "tuple_id": self.tuple_id,
            "condition": self.condition,
            "split": self.split,
            "tuple_labels": dict(self.tuple_labels),
            "meta": dict(self.meta),
            "attempt": self.attempt,
            "is_replacement": self.is_replacement,
            "replaced_from": self.replaced_from,
            "rep_index": self.rep_index,
            "status": self.status,
            "gen_batch_id": self.gen_batch_id,
            "ver_batch_id": self.ver_batch_id,
            "text": self.text,
            "generator": self.generator,
            "verifier_labels": self.verifier_labels,
            "verifier": self.verifier,
            "reject_reasons": list(self.reject_reasons),
            "requeues": [dict(r) for r in self.requeues],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TargetItem:
        return cls(
            item_id=d["item_id"],
            tuple_id=d["tuple_id"],
            condition=d["condition"],
            split=d["split"],
            tuple_labels=d["tuple_labels"],
            meta=d.get("meta", {}),
            attempt=d.get("attempt", 1),
            is_replacement=d.get("is_replacement", False),
            replaced_from=d.get("replaced_from"),
            rep_index=d.get("rep_index", 0),
            status=d.get("status", "pending_gen"),
            gen_batch_id=d.get("gen_batch_id"),
            ver_batch_id=d.get("ver_batch_id"),
            text=d.get("text"),
            generator=d.get("generator"),
            verifier_labels=d.get("verifier_labels"),
            verifier=d.get("verifier"),
            reject_reasons=d.get("reject_reasons", []),
            requeues=d.get("requeues", []),
        )

    def to_case(self, active_service_options: dict[str, str] | None = None) -> Case:
        meta = dict(self.meta)
        meta["attempts"] = self.attempt
        meta["is_replacement"] = self.is_replacement
        if self.replaced_from:
            meta["replaced_from"] = self.replaced_from
        meta["generator"] = self.generator or ""
        meta["verifier"] = self.verifier or ""
        meta["generator_model"] = self.generator or ""
        meta["verifier_model"] = self.verifier or ""

        return Case(
            case_id=self.tuple_id,
            text=self.text or "",
            fields=list(self.tuple_labels.keys()),
            service_options=active_service_options,
            bundle_size=1,
            truth=[dict(self.tuple_labels)],
            meta=meta,
        )


class CorpusExchange:
    """Manages the file-exchange transport protocol for corpus generation and verification."""

    def __init__(
        self,
        exchange_dir: str | Path = "runs/_corpus_exchange",
        data_dir: str | Path = "data/edgebench/v1",
        seed: int = 20260924,
    ) -> None:
        self.exchange_dir = Path(exchange_dir)
        self.data_dir = Path(data_dir)
        self.seed = seed

        self.gen_todo_dir = self.exchange_dir / "gen" / "todo"
        self.gen_done_dir = self.exchange_dir / "gen" / "done"
        self.ver_todo_dir = self.exchange_dir / "ver" / "todo"
        self.ver_done_dir = self.exchange_dir / "ver" / "done"
        self.targets_dir = self.exchange_dir / "_targets"

        for d in (
            self.gen_todo_dir,
            self.gen_done_dir,
            self.ver_todo_dir,
            self.ver_done_dir,
            self.targets_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

        self.items: dict[str, TargetItem] = {}
        self.ingested_gen_files: set[str] = set()
        self.ingested_ver_files: set[str] = set()
        # Ver batch IDs moved to ver/_superseded/ by requeue_ver: never ingested, never reused
        self.superseded_ver_batches: set[str] = set()
        # Gen batch IDs moved to gen/_superseded/ by requeue_gen: never ingested, never reused
        self.superseded_gen_batches: set[str] = set()
        self._load_state()

    def _state_file(self) -> Path:
        return self.targets_dir / "state.json"

    def _load_state(self) -> None:
        state_file = self._state_file()
        if state_file.exists():
            # Fail closed: a state file that does not load is an error, never a silent reset.
            try:
                with open(state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item_dict in data.get("items", []):
                        item = TargetItem.from_dict(item_dict)
                        self.items[item.item_id] = item
                    self.ingested_gen_files = set(data.get("ingested_gen_files", []))
                    self.ingested_ver_files = set(data.get("ingested_ver_files", []))
                    self.superseded_ver_batches = set(data.get("superseded_ver_batches", []))
                    self.superseded_gen_batches = set(data.get("superseded_gen_batches", []))
            except Exception as e:
                raise ValueError(f"Corrupt exchange state {state_file}: {e}") from e

    def _save_state(self) -> None:
        state_file = self._state_file()
        data = {
            "items": [it.to_dict() for it in self.items.values()],
            "ingested_gen_files": sorted(self.ingested_gen_files),
            "ingested_ver_files": sorted(self.ingested_ver_files),
            "superseded_ver_batches": sorted(self.superseded_ver_batches),
            "superseded_gen_batches": sorted(self.superseded_gen_batches),
        }
        tmp_file = state_file.with_suffix(".tmp")
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp_file.replace(state_file)

    def _in_frozen_group(self, it: TargetItem, frozen: dict[str, Any]) -> bool:
        return group_key(it.condition, it.split) in frozen

    def _refuse_frozen_changes(self, op: str, item_ids: list[str]) -> None:
        """Raise (nothing changed, state not saved) if any of these items belongs to a frozen group."""
        frozen = load_frozen(self.data_dir)
        hit = sorted(i for i in set(item_ids) if i in self.items and self._in_frozen_group(self.items[i], frozen))
        if hit:
            groups = sorted({group_key(self.items[i].condition, self.items[i].split) for i in hit})
            raise ValueError(
                f"{op}: refusing to change {len(hit)} item(s) of frozen group(s) {groups}: {hit}; nothing changed"
            )

    def _pending_done_ids(self, done_dir: Path, ingested: set[str], skip_statuses: tuple[str, ...]) -> list[str]:
        """Item IDs of uningested done files that ingest would change (lenient read; ingest validates)."""
        ids: list[str] = []
        for done_f in sorted(done_dir.glob("*.jsonl")):
            if done_f.name in ingested or done_f.stem in self.superseded_ver_batches | self.superseded_gen_batches:
                continue
            with open(done_f, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        iid = json.loads(line)["item_id"].strip()
                    except Exception:
                        continue
                    if iid in self.items and self.items[iid].status not in skip_statuses:
                        ids.append(iid)
        return ids

    def _get_active_service_options(
        self, condition: str, catalogs: dict[str, list[str]], all_svcs: dict[str, Any]
    ) -> dict[str, str] | None:
        if not is_rq4_condition(condition):
            return None
        norm_cond = condition.split("/")[-1]

        active_cat_ids: list[str] = []
        if norm_cond in ("K4", "RQ4_K4"):
            active_cat_ids = catalogs["C_4"]
        elif norm_cond in ("K15", "RQ4_K15"):
            active_cat_ids = catalogs["C_15"]
        elif norm_cond in ("K64", "RQ4_K64", "churn0", "RQ4_churn0"):
            active_cat_ids = catalogs["C_64"]
        elif norm_cond in ("K128", "RQ4_K128"):
            active_cat_ids = catalogs["C_128"]
        elif norm_cond in ("K254", "RQ4_K254"):
            active_cat_ids = catalogs["C_254"]
        elif norm_cond in ("churn25", "RQ4_churn25"):
            active_cat_ids = catalogs["v25"]
        elif norm_cond in ("churn50", "RQ4_churn50"):
            active_cat_ids = catalogs["v50"]
        else:
            return None

        opts: dict[str, str] = {sid: all_svcs[sid]["description"] for sid in active_cat_ids}
        opts["unsupported"] = "unsupported service"
        return opts

    def init_targets(
        self,
        conditions: list[str] | None = None,
        splits: list[str] | None = None,
        n_test: int = 300,
        n_dev: int = 60,
    ) -> None:
        """Initialize target tuples for the requested conditions and splits if not already present."""
        cond_list = conditions or ALL_CONDITIONS
        split_list = splits or ["test", "dev"]

        cat_dir = self.data_dir / "catalog"
        if not (cat_dir / "catalogs.json").exists():
            save_catalogs(cat_dir, seed=self.seed)

        catalogs = load_catalogs(cat_dir)
        services_data = load_services(cat_dir)
        novel_ids = [s["id"] for s in services_data["novel_services"]]

        existing_tuples: set[tuple[str, str, str]] = {
            (it.condition, it.split, it.tuple_id) for it in self.items.values()
        }

        new_items_added = False
        for cond in cond_list:
            for sp in split_list:
                n = n_test if sp == "test" else n_dev

                # Check if already initialized
                existing_for_cond = [
                    it for it in self.items.values()
                    if it.condition == cond and it.split == sp and not it.is_replacement
                ]
                if len(existing_for_cond) >= n:
                    continue

                # Load from _tuples if available, else sample
                tuple_file = self.data_dir / "_tuples" / f"{cond}_{sp}.jsonl"
                tuples: list[TupleItem] = []
                if tuple_file.exists():
                    with open(tuple_file, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                d = json.loads(line)
                                tuples.append(
                                    TupleItem(
                                        condition=d["condition"],
                                        split=d["split"],
                                        tuple_id=d["tuple_id"],
                                        tuple_labels=d["tuple_labels"],
                                        meta=d.get("meta", {}),
                                    )
                                )

                if len(tuples) < n:
                    tuples = sample_tuples_for_condition(
                        condition=cond,
                        split=sp,
                        n=n,
                        base_seed=self.seed,
                        catalogs=catalogs,
                        novel_service_ids=novel_ids,
                    )

                for t in tuples:
                    if (cond, sp, t.tuple_id) in existing_tuples:
                        continue
                    item_id = make_opaque_item_id(t.tuple_id, cond, sp, attempt=1, rep_index=0)
                    t_item = TargetItem(
                        item_id=item_id,
                        tuple_id=t.tuple_id,
                        condition=cond,
                        split=sp,
                        tuple_labels=dict(t.tuple_labels),
                        meta=dict(t.meta),
                        attempt=1,
                        is_replacement=False,
                        status="pending_gen",
                    )
                    self.items[item_id] = t_item
                    existing_tuples.add((cond, sp, t.tuple_id))
                    new_items_added = True

        if new_items_added:
            self._save_state()

    def export_gen(
        self,
        conditions: list[str] | None = None,
        splits: list[str] | None = None,
        gen_batch_size: int = 100,
        n_test: int = 300,
        n_dev: int = 60,
    ) -> list[Path]:
        """Export todo batches for everything not yet accepted or needing generation retry.

        Idempotent: never re-exports an item that already has a done line or is already in an active todo batch.
        Items of a frozen (condition, split) are never exported.
        """
        self.init_targets(conditions=conditions, splits=splits, n_test=n_test, n_dev=n_dev)

        cat_dir = self.data_dir / "catalog"
        services_data = load_services(cat_dir)
        service_descs = {
            s["id"]: s["description"]
            for s in services_data["services"] + services_data["novel_services"]
        }

        # Collect items that have already been written to an active (uningested) gen/done file
        active_done_item_ids: set[str] = set()
        for done_f in self.gen_done_dir.glob("*.jsonl"):
            if done_f.name in self.ingested_gen_files or done_f.stem in self.superseded_gen_batches:
                continue
            try:
                with open(done_f, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            d = json.loads(line)
                            if "item_id" in d:
                                active_done_item_ids.add(d["item_id"])
            except Exception:
                pass

        # Collect items that are already in an active (uningested) gen/todo file
        active_todo_item_ids: set[str] = set()
        for todo_f in self.gen_todo_dir.glob("*.jsonl"):
            if todo_f.name in self.ingested_gen_files or todo_f.stem in self.superseded_gen_batches:
                continue
            try:
                with open(todo_f, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            d = json.loads(line)
                            if "item_id" in d:
                                active_todo_item_ids.add(d["item_id"])
            except Exception:
                pass

        cond_list = conditions or ALL_CONDITIONS
        split_list = splits or ["test", "dev"]
        frozen = load_frozen(self.data_dir)

        # Select items needing generation
        pending_items: list[TargetItem] = []
        for it in self.items.values():
            if it.condition not in cond_list or it.split not in split_list:
                continue
            if self._in_frozen_group(it, frozen):
                continue
            if it.status == "accepted":
                continue
            # Never re-export an item that already has an active done line
            if it.item_id in active_done_item_ids:
                continue
            # Never re-export an item already in an active todo file
            if it.item_id in active_todo_item_ids:
                continue

            # "rejected" is terminal (a replacement item carries the tuple on), so it is never re-exported.
            if it.status == "pending_gen":
                pending_items.append(it)
            elif it.status == "exported_gen":
                # Its todo file is gone or its batch was ingested without it; re-queue
                pending_items.append(it)

        if not pending_items:
            return []

        # Group by (condition, split)
        grouped: dict[tuple[str, str], list[TargetItem]] = defaultdict(list)
        for it in pending_items:
            grouped[(it.condition, it.split)].append(it)

        exported_files: list[Path] = []

        for (cond, sp), items_group in sorted(grouped.items()):
            # Chunk into batches of gen_batch_size
            for chunk_idx in range(0, len(items_group), gen_batch_size):
                batch_items = items_group[chunk_idx : chunk_idx + gen_batch_size]

                # Find unique batch_id
                batch_num = 1
                while True:
                    batch_id = f"gen_{cond}_{sp}_{batch_num:03d}"
                    target_file = self.gen_todo_dir / f"{batch_id}.jsonl"
                    if not target_file.exists() and batch_id not in self.superseded_gen_batches:
                        break
                    batch_num += 1

                lines_to_write: list[str] = []
                for it in batch_items:
                    t_item = TupleItem(
                        condition=it.condition,
                        split=it.split,
                        tuple_id=it.tuple_id,
                        tuple_labels=it.tuple_labels,
                        meta=it.meta,
                    )
                    prompt_text = build_single_item_generator_prompt(t_item, service_descs)
                    it.meta["prompt_sha256"] = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
                    wording = it.meta.get("wording_family", "")
                    max_words = 120 if wording in ("email", "formal SLA clause") else 80

                    record = {
                        "item_id": it.item_id,
                        "condition": it.condition,
                        "split": it.split,
                        "prompt": prompt_text,
                        "max_words": max_words,
                    }
                    lines_to_write.append(json.dumps(record))
                    it.status = "exported_gen"
                    it.gen_batch_id = batch_id

                with open(target_file, "w", encoding="utf-8") as f:
                    for l in lines_to_write:
                        f.write(l + "\n")

                exported_files.append(target_file)

        self._save_state()
        return exported_files

    def ingest_gen(self) -> dict[str, Any]:
        """Validate gen done files and run static lints (scaffolds, catalog leak, 4-gram copy).

        Rejected items receive an incremented attempt counter or trigger the replacement policy.
        Refuses (raises before any change) if it would change an item of a frozen (condition, split).
        """
        cat_dir = self.data_dir / "catalog"
        services_data = load_services(cat_dir)
        all_svcs = {
            s["id"]: s for s in services_data["services"] + services_data["novel_services"]
        }
        service_descs = {sid: s["description"] for sid, s in all_svcs.items()}

        done_files = sorted(self.gen_done_dir.glob("*.jsonl"))
        if not done_files:
            return {"status": "no_done_files", "processed": 0}

        self._refuse_frozen_changes(
            "ingest_gen",
            self._pending_done_ids(self.gen_done_dir, self.ingested_gen_files, ("accepted", "done_gen", "exported_ver")),
        )

        results: dict[str, Any] = {
            "files_processed": len(done_files),
            "items_passed": 0,
            "items_rejected": 0,
            "rejections": Counter(),
            "superseded_skipped": [],
        }

        for done_file in done_files:
            # A done file for a batch requeue_gen superseded (a generator finished it late) is never ingested
            if done_file.stem in self.superseded_gen_batches:
                results["superseded_skipped"].append(done_file.name)
                continue
            if done_file.name in self.ingested_gen_files:
                continue

            # 1. Validation
            records: list[dict[str, Any]] = []
            seen_ids: set[str] = set()

            with open(done_file, "r", encoding="utf-8") as f:
                line_idx = 0
                for line in f:
                    line_idx += 1
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception as e:
                        raise ValueError(f"Malformed gen done file {done_file.name} line {line_idx}: invalid JSON: {e}")

                    if not isinstance(obj, dict):
                        raise ValueError(f"Malformed gen done file {done_file.name} line {line_idx}: expected JSON object")

                    for req_key in ("item_id", "text", "generator"):
                        if req_key not in obj:
                            raise ValueError(f"Malformed gen done file {done_file.name} line {line_idx}: missing '{req_key}'")
                        if not isinstance(obj[req_key], str) or not obj[req_key].strip():
                            raise ValueError(f"Malformed gen done file {done_file.name} line {line_idx}: '{req_key}' must be non-empty string")

                    item_id = obj["item_id"].strip()
                    if item_id in seen_ids:
                        raise ValueError(f"Malformed gen done file {done_file.name}: duplicate item_id '{item_id}'")
                    seen_ids.add(item_id)

                    if item_id not in self.items:
                        raise ValueError(f"Malformed gen done file {done_file.name}: unknown item_id '{item_id}'")

                    records.append(obj)

            if not records:
                raise ValueError(f"Malformed gen done file {done_file.name}: file is empty")

            _check_done_matches_todo("gen", done_file, self.gen_todo_dir / done_file.name, seen_ids)

            # 2. Run static lints on each item
            for rec in records:
                item_id = rec["item_id"]
                text = rec["text"].strip()
                generator_str = rec["generator"].strip()
                it = self.items[item_id]

                # If already accepted or ready for ver, skip
                if it.status in ("accepted", "done_gen", "exported_ver"):
                    continue

                is_rq4 = is_rq4_condition(it.condition)

                # Static Lint 1: Forbidden scaffolds
                sc_ok, _ = check_scaffold_lint(text)
                if not sc_ok:
                    self._reject_item(it, "lint_scaffold")
                    results["items_rejected"] += 1
                    results["rejections"]["lint_scaffold"] += 1
                    continue

                target_svc_id = it.meta.get("target_service", it.tuple_labels.get("service_type"))

                # Static Lint 2: Enum / field-name leak (all conditions; RQ4 adds the target service id and name)
                enum_ok, _ = check_enum_leak_lint(
                    text,
                    target_service_id=target_svc_id if is_rq4 else None,
                    target_service_name=all_svcs.get(target_svc_id, {}).get("name") if is_rq4 else None,
                )
                if not enum_ok:
                    self._reject_item(it, "lint_enum_leak")
                    results["items_rejected"] += 1
                    results["rejections"]["lint_enum_leak"] += 1
                    continue

                # Static Lint 3: Code-switch texts must actually mix English and the target language
                if "codeswitch" in it.condition:
                    cs_ok, _ = check_codeswitch_monolingual_lint(text, it.meta.get("language", "es"))
                    if not cs_ok:
                        self._reject_item(it, "lint_codeswitch_monolingual")
                        results["items_rejected"] += 1
                        results["rejections"]["lint_codeswitch_monolingual"] += 1
                        continue

                # Static Lint 4: Catalog leak for non-RQ4
                if not is_rq4:
                    leak_ok, _ = check_catalog_leak_lint(text)
                    if not leak_ok:
                        self._reject_item(it, "lint_catalog_leak")
                        results["items_rejected"] += 1
                        results["rejections"]["lint_catalog_leak"] += 1
                        continue

                # Static Lint 5: Verbatim 4-gram copy. For RQ4 the service source is only the target's own
                # description (passed below), so the verdict is identical for every catalog size K.
                sample_fields = list(it.tuple_labels.keys())
                active_criteria_descs: list[str] = []
                for f in sample_fields:
                    if f == "service_type":
                        if not is_rq4:
                            active_criteria_descs.extend(DEFAULT_CRITERIA["service_type"].values())
                    else:
                        active_criteria_descs.extend(DEFAULT_CRITERIA[f].values())

                target_svc_desc = service_descs.get(target_svc_id, "") if is_rq4 else ""
                cp_ok, _ = check_copy_4gram_lint(text, active_criteria_descs, target_svc_desc)
                if not cp_ok:
                    self._reject_item(it, "lint_copy_4gram")
                    results["items_rejected"] += 1
                    results["rejections"]["lint_copy_4gram"] += 1
                    continue

                # All static lints passed
                it.text = text
                it.generator = generator_str
                it.status = "done_gen"
                results["items_passed"] += 1

            self.ingested_gen_files.add(done_file.name)
            self._save_state()

        return results

    def _reject_item(self, it: TargetItem, reason: str) -> None:
        """Handle rejection: retry if attempts < 3, else apply replacement policy."""
        it.reject_reasons.append(reason)
        if it.attempt < 3:
            it.attempt += 1
            it.status = "pending_gen"
            it.gen_batch_id = None
            it.ver_batch_id = None
            it.text = None
            it.generator = None
            it.verifier_labels = None
            it.verifier = None
        else:
            # Replacement policy: mark current item rejected, create replacement item
            it.status = "rejected"
            rep_idx = it.rep_index + 1
            new_id = make_opaque_item_id(it.tuple_id, it.condition, it.split, attempt=1, rep_index=rep_idx)
            rep_meta = dict(it.meta)
            rep_meta["replaced_from"] = it.item_id
            rep_item = TargetItem(
                item_id=new_id,
                tuple_id=it.tuple_id,
                condition=it.condition,
                split=it.split,
                tuple_labels=dict(it.tuple_labels),
                meta=rep_meta,
                attempt=1,
                is_replacement=True,
                replaced_from=it.item_id,
                rep_index=rep_idx,
                status="pending_gen",
            )
            self.items[new_id] = rep_item

    def export_ver(
        self,
        conditions: list[str] | None = None,
        splits: list[str] | None = None,
        ver_batch_size: int = 100,
    ) -> list[Path]:
        """Export ver todo batches and context files for generated items.

        CRITICAL: ver files contain ONLY item_id and text.
        Context markdown contains field instructions & catalog (for RQ4).
        NO target tuple and NO condition name appears in the ver files. Items of a frozen (condition, split)
        are never exported.
        Pending items are shuffled with a seeded RNG and packed so that no batch holds two items with
        the same tuple_group_key (parallel texts of one tuple would let a verifier copy labels across).
        """
        cat_dir = self.data_dir / "catalog"
        catalogs = load_catalogs(cat_dir)
        services_data = load_services(cat_dir)
        all_svcs = {
            s["id"]: s for s in services_data["services"] + services_data["novel_services"]
        }

        # Check uningested done IDs
        done_ver_ids: set[str] = set()
        for done_f in self.ver_done_dir.glob("*.jsonl"):
            if done_f.name in self.ingested_ver_files or done_f.stem in self.superseded_ver_batches:
                continue
            try:
                with open(done_f, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            d = json.loads(line)
                            if "item_id" in d:
                                done_ver_ids.add(d["item_id"])
            except Exception:
                pass

        # Check uningested todo IDs
        todo_ver_ids: set[str] = set()
        for todo_f in self.ver_todo_dir.glob("*.jsonl"):
            if todo_f.name in self.ingested_ver_files or todo_f.stem in self.superseded_ver_batches:
                continue
            try:
                with open(todo_f, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            d = json.loads(line)
                            if "item_id" in d:
                                todo_ver_ids.add(d["item_id"])
            except Exception:
                pass

        cond_list = conditions or ALL_CONDITIONS
        split_list = splits or ["test", "dev"]
        frozen = load_frozen(self.data_dir)

        pending_ver: list[TargetItem] = []
        for it in self.items.values():
            if it.condition not in cond_list or it.split not in split_list:
                continue
            if self._in_frozen_group(it, frozen):
                continue
            # exported_ver items not in any active todo/done batch (their batch was ingested without them) are re-queued
            if it.status not in ("done_gen", "exported_ver"):
                continue
            if not it.text:
                continue
            if it.item_id in done_ver_ids or it.item_id in todo_ver_ids:
                continue
            pending_ver.append(it)

        if not pending_ver:
            return []

        # Seeded shuffle (corpus seed + stable salt) so batches do not follow state order
        salt = f"{self.seed}:export_ver:tuple_group_isolation"
        rng = random.Random(int(hashlib.sha256(salt.encode("utf-8")).hexdigest()[:16], 16))
        pending_ver.sort(key=lambda it: it.item_id)
        rng.shuffle(pending_ver)

        # Group items by verification context (fields + catalog)
        # For non-RQ4, grouped by field length (F4, F6, F8); for RQ4, grouped by condition so catalog is shared
        def ver_context_key(it: TargetItem) -> tuple[int, str]:
            norm = it.condition.split("/")[-1]
            if is_rq4_condition(it.condition):
                return (len(it.tuple_labels), norm)
            return (len(it.tuple_labels), "standard")

        grouped: dict[tuple[int, str], list[TargetItem]] = defaultdict(list)
        for it in pending_ver:
            grouped[ver_context_key(it)].append(it)

        exported_files: list[Path] = []

        for (_, ctx_name), items_group in sorted(grouped.items()):
            # First-fit packing: an item joins the first batch with room and no item of its tuple group
            packed: list[tuple[list[TargetItem], set[tuple[Any, ...]]]] = []
            for it in items_group:
                key = tuple_group_key(it)
                for b_items, b_keys in packed:
                    if len(b_items) < ver_batch_size and key not in b_keys:
                        b_items.append(it)
                        b_keys.add(key)
                        break
                else:
                    packed.append(([it], {key}))

            for batch_items, _ in packed:
                # Find unique opaque batch ID (no condition names!); superseded IDs are never reused
                batch_num = 1
                while True:
                    batch_id = f"vbatch_{batch_num:04d}"
                    target_jsonl = self.ver_todo_dir / f"{batch_id}.jsonl"
                    target_ctx = self.ver_todo_dir / f"{batch_id}.context.md"
                    if (
                        not target_jsonl.exists()
                        and not target_ctx.exists()
                        and batch_id not in self.superseded_ver_batches
                    ):
                        break
                    batch_num += 1

                # 1. Write ver/todo/<batch_id>.jsonl (ONLY item_id and text)
                with open(target_jsonl, "w", encoding="utf-8") as f:
                    for it in batch_items:
                        line_obj = {"item_id": it.item_id, "text": it.text}
                        f.write(json.dumps(line_obj) + "\n")
                        it.status = "exported_ver"
                        it.ver_batch_id = batch_id

                # 2. Write ver/todo/<batch_id>.context.md
                sample_item = batch_items[0]
                sample_fields = list(sample_item.tuple_labels.keys())
                active_opts = self._get_active_service_options(sample_item.condition, catalogs, all_svcs)
                context_md = build_verifier_context_markdown(sample_fields, active_opts)
                with open(target_ctx, "w", encoding="utf-8") as f:
                    f.write(context_md)

                exported_files.extend([target_jsonl, target_ctx])

        self._save_state()
        return exported_files

    def ingest_ver(self) -> dict[str, Any]:
        """Validate ver done files and run acceptance logic (label match & incremental diversity).

        Refuses (raises before any change) if it would change an item of a frozen (condition, split).
        """
        done_files = sorted(self.ver_done_dir.glob("*.jsonl"))
        if not done_files:
            return {"status": "no_done_files", "processed": 0}

        results: dict[str, Any] = {
            "files_processed": len(done_files),
            "items_accepted": 0,
            "items_rejected": 0,
            "rejections": Counter(),
            "superseded_skipped": [],
        }
        self._refuse_frozen_changes(
            "ingest_ver", self._pending_done_ids(self.ver_done_dir, self.ingested_ver_files, ("accepted",))
        )

        for done_file in done_files:
            # A done file for a batch requeue_ver superseded (a verifier finished it late) is never ingested
            if done_file.stem in self.superseded_ver_batches:
                results["superseded_skipped"].append(done_file.name)
                continue
            if done_file.name in self.ingested_ver_files:
                continue
            records: list[dict[str, Any]] = []
            seen_ids: set[str] = set()

            with open(done_file, "r", encoding="utf-8") as f:
                line_idx = 0
                for line in f:
                    line_idx += 1
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception as e:
                        raise ValueError(f"Malformed ver done file {done_file.name} line {line_idx}: invalid JSON: {e}")

                    if not isinstance(obj, dict):
                        raise ValueError(f"Malformed ver done file {done_file.name} line {line_idx}: expected JSON object")

                    for req_key in ("item_id", "labels", "verifier"):
                        if req_key not in obj:
                            raise ValueError(f"Malformed ver done file {done_file.name} line {line_idx}: missing '{req_key}'")

                    if not isinstance(obj["item_id"], str) or not obj["item_id"].strip():
                        raise ValueError(f"Malformed ver done file {done_file.name} line {line_idx}: 'item_id' must be non-empty string")
                    if not isinstance(obj["labels"], dict):
                        raise ValueError(f"Malformed ver done file {done_file.name} line {line_idx}: 'labels' must be a dict")
                    if not isinstance(obj["verifier"], str) or not obj["verifier"].strip():
                        raise ValueError(f"Malformed ver done file {done_file.name} line {line_idx}: 'verifier' must be non-empty string")

                    item_id = obj["item_id"].strip()
                    if item_id in seen_ids:
                        raise ValueError(f"Malformed ver done file {done_file.name}: duplicate item_id '{item_id}'")
                    seen_ids.add(item_id)

                    if item_id not in self.items:
                        raise ValueError(f"Malformed ver done file {done_file.name}: unknown item_id '{item_id}'")

                    records.append(obj)

            if not records:
                raise ValueError(f"Malformed ver done file {done_file.name}: file is empty")

            _check_done_matches_todo("ver", done_file, self.ver_todo_dir / done_file.name, seen_ids)

            # Verify labels and diversity for each record
            for rec in records:
                item_id = rec["item_id"]
                labels = rec["labels"]
                verifier_str = rec["verifier"].strip()
                it = self.items[item_id]

                if it.status == "accepted":
                    continue

                # 1. Label match check
                label_match = True
                for f_name, target_val in it.tuple_labels.items():
                    if labels.get(f_name) != target_val:
                        label_match = False
                        break

                if not label_match:
                    self._reject_item(it, "verifier_mismatch")
                    results["items_rejected"] += 1
                    results["rejections"]["verifier_mismatch"] += 1
                    continue

                # 2. Incremental diversity check against currently accepted items for this condition & split
                accepted_for_cond = [
                    x for x in self.items.values()
                    if x.condition == it.condition and x.split == it.split and x.status == "accepted"
                ]
                total_target_n = sum(
                    1 for x in self.items.values()
                    if x.condition == it.condition and x.split == it.split and not x.is_replacement
                )
                max_opening_count = max(1, int(total_target_n * 0.05))
                max_5gram_count = max(1, int(total_target_n * 0.10))

                words = normalize_words(it.text or "")
                div_ok = True
                div_reason = ""
                if len(words) >= 4:
                    op = tuple(words[:4])
                    cur_op_cnt = sum(
                        1 for c in accepted_for_cond
                        if len(normalize_words(c.text or "")) >= 4 and tuple(normalize_words(c.text or "")[:4]) == op
                    )
                    if cur_op_cnt + 1 > max_opening_count:
                        div_ok = False
                        div_reason = "lint_diversity_opening_4gram"

                if div_ok:
                    text_5grams = extract_ngrams(words, 5)
                    for g in text_5grams:
                        cur_g_cnt = sum(
                            1 for c in accepted_for_cond
                            if g in extract_ngrams(normalize_words(c.text or ""), 5)
                        )
                        if cur_g_cnt + 1 > max_5gram_count:
                            div_ok = False
                            div_reason = "lint_diversity_5gram"
                            break

                if not div_ok:
                    self._reject_item(it, div_reason)
                    results["items_rejected"] += 1
                    results["rejections"][div_reason] += 1
                    continue

                # Accepted (state only; the formal corpus files are written by assemble())
                it.status = "accepted"
                it.verifier_labels = labels
                it.verifier = verifier_str
                results["items_accepted"] += 1

            self.ingested_ver_files.add(done_file.name)
            self._save_state()

        return results

    def requeue_ver(
        self,
        batch_ids: list[str],
        item_ids: list[str] | None = None,
        reason: str = "requeue-ver",
    ) -> dict[str, Any]:
        """Send items verified or exported in the given ver batches back to verification.

        Items of those batches with status exported_ver or accepted (assemble has not run) return to
        done_gen with text, generator and attempt unchanged and verifier labels cleared; each records
        {requeued_from, reason, archive} in `requeues`. item_ids, if given, narrows the requeue to those items
        (they must belong to the named batches). The batches' ver/todo (.jsonl, .context.md) and ver/done
        files move to ver/_superseded/<timestamp>/{todo,done}/ (never deleted), and the batch IDs are
        recorded as superseded: ingest_ver skips them and export_ver never reuses them. Items no longer
        in the batch (rejected by the verifier and back in generation) are reported as skipped.
        Validation runs before any change; nothing moves if it fails. Items of a frozen (condition, split)
        are never requeued: the call raises instead.
        """
        batch_ids = list(dict.fromkeys(batch_ids))
        if not batch_ids:
            raise ValueError("requeue_ver: no batch IDs given")
        members: dict[str, list[str]] = {}
        for bid in batch_ids:
            if bid in self.superseded_ver_batches:
                raise ValueError(f"requeue_ver: batch {bid} is already superseded")
            todo_file = self.ver_todo_dir / f"{bid}.jsonl"
            if not todo_file.exists():
                raise ValueError(f"requeue_ver: no ver todo batch {todo_file.name}")
            ids: list[str] = []
            with open(todo_file, "r", encoding="utf-8") as f:
                for l in f:
                    if l.strip():
                        ids.append(json.loads(l)["item_id"])
            ids.extend(i for i, it in self.items.items() if it.ver_batch_id == bid and i not in ids)
            members[bid] = ids

        in_batches = {i for ids in members.values() for i in ids}
        if item_ids is not None:
            outside = sorted(set(item_ids) - in_batches)
            if outside:
                raise ValueError(f"requeue_ver: item IDs not in the named batches: {outside}")
            selected = set(item_ids)
        else:
            selected = in_batches
        self._refuse_frozen_changes("requeue_ver", [
            i for bid, ids in members.items() for i in ids
            if i in selected and i in self.items and self.items[i].ver_batch_id == bid
            and self.items[i].status in ("exported_ver", "accepted")
        ])

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archive = self.exchange_dir / "ver" / "_superseded" / stamp
        n = 1
        while archive.exists():
            n += 1
            archive = self.exchange_dir / "ver" / "_superseded" / f"{stamp}_{n}"

        requeued: list[str] = []
        skipped: list[str] = []
        for bid, ids in members.items():
            for i in ids:
                if i not in selected:
                    continue
                it = self.items.get(i)
                if it is None or it.ver_batch_id != bid or it.status not in ("exported_ver", "accepted"):
                    skipped.append(i)
                    continue
                it.status = "done_gen"
                it.ver_batch_id = None
                it.verifier_labels = None
                it.verifier = None
                it.requeues.append({"requeued_from": bid, "reason": reason, "archive": archive.name})
                requeued.append(i)
        self.superseded_ver_batches.update(batch_ids)
        self._save_state()

        # State first: files left behind by an interruption belong to superseded batches and are ignored
        for bid in batch_ids:
            for src, sub in (
                (self.ver_todo_dir / f"{bid}.jsonl", "todo"),
                (self.ver_todo_dir / f"{bid}.context.md", "todo"),
                (self.ver_done_dir / f"{bid}.jsonl", "done"),
            ):
                if src.exists():
                    (archive / sub).mkdir(parents=True, exist_ok=True)
                    src.replace(archive / sub / src.name)

        return {"requeued": requeued, "skipped": skipped, "archive_dir": str(archive)}

    def _archive_dir(self, stage: str) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archive = self.exchange_dir / stage / "_superseded" / stamp
        n = 1
        while archive.exists():
            n += 1
            archive = self.exchange_dir / stage / "_superseded" / f"{stamp}_{n}"
        return archive

    def _active_batch_files(self) -> dict[str, list[str]]:
        """item_id -> active (uningested, not superseded) gen/ver todo and done files that list it."""
        refs: dict[str, list[str]] = defaultdict(list)
        for stage, dirs, ingested, superseded in (
            ("gen", (self.gen_todo_dir, self.gen_done_dir), self.ingested_gen_files, self.superseded_gen_batches),
            ("ver", (self.ver_todo_dir, self.ver_done_dir), self.ingested_ver_files, self.superseded_ver_batches),
        ):
            for d in dirs:
                for f in sorted(d.glob("*.jsonl")):
                    if f.name in ingested or f.stem in superseded:
                        continue
                    with open(f, "r", encoding="utf-8") as fh:
                        for line in fh:
                            try:
                                refs[json.loads(line)["item_id"].strip()].append(f"{stage}/{d.name}/{f.name}")
                            except Exception:
                                continue
        return refs

    def retarget(
        self,
        item_ids: list[str],
        allow_accepted: bool = False,
        reason: str = "retarget",
    ) -> dict[str, Any]:
        """Give RQ4 unsupported items a new unsupported target and send them back to generation.

        The new target comes from the condition's unsupported pool (rq4_active_catalog_and_unsupported_pool
        over the current catalog), chosen with an RNG seeded by (corpus seed, item_id); targets already used
        by another unsupported item of the same (condition, split), or chosen earlier in this call, are
        avoided while the pool has others. meta.target_service changes (tuple_labels.service_type stays
        "unsupported"); text, generator, verifier fields and batch IDs are cleared, status becomes
        pending_gen, and {op, retargeted_from, retargeted_to, prior_status, reason, time} is appended to
        `requeues`. Refused, with nothing changed: unknown or non-RQ4 items, supported items, rejected
        items, accepted items without allow_accepted, items of a frozen (condition, split), and items listed
        in an active gen/ver todo or done file (run requeue_gen / requeue_ver_items or ingest first).
        """
        item_ids = sorted(set(item_ids))
        if not item_ids:
            raise ValueError("retarget: no item IDs given")
        unknown = [i for i in item_ids if i not in self.items]
        if unknown:
            raise ValueError(f"retarget: unknown item IDs: {unknown}")
        problems: list[str] = []
        refs = self._active_batch_files()
        for i in item_ids:
            it = self.items[i]
            if not is_rq4_condition(it.condition):
                problems.append(f"{i}: not an RQ4 item ({it.condition})")
            elif it.tuple_labels.get("service_type") != "unsupported":
                problems.append(f"{i}: not an unsupported item (service_type={it.tuple_labels.get('service_type')})")
            elif it.status == "rejected":
                problems.append(f"{i}: rejected (terminal)")
            elif it.status == "accepted" and not allow_accepted:
                problems.append(f"{i}: accepted (pass allow_accepted to retarget it)")
            elif refs.get(i):
                problems.append(f"{i}: listed in active batch file(s) {refs[i]}")
        if problems:
            raise ValueError("retarget: refusing, nothing changed:\n  " + "\n  ".join(problems))
        self._refuse_frozen_changes("retarget", item_ids)

        cat_dir = self.data_dir / "catalog"
        catalogs = load_catalogs(cat_dir)
        novel_ids = [s["id"] for s in load_services(cat_dir)["novel_services"]]
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        chosen: dict[str, dict[str, str]] = {}
        picked: dict[tuple[str, str], set[str]] = defaultdict(set)  # new targets chosen earlier in this call
        for i in item_ids:
            it = self.items[i]
            old = it.meta.get("target_service")
            _, pool = rq4_active_catalog_and_unsupported_pool(it.condition, catalogs, novel_ids)
            used = picked[(it.condition, it.split)] | {
                x.meta.get("target_service") for x in self.items.values()
                if x.item_id != i and x.condition == it.condition and x.split == it.split
                and x.status != "rejected" and x.tuple_labels.get("service_type") == "unsupported"
            }
            candidates = sorted(s for s in pool if s != old)
            if not candidates:
                raise ValueError(f"retarget: empty unsupported pool for {i} ({it.condition}); nothing changed")
            unused = [s for s in candidates if s not in used]
            rng = random.Random(int(hashlib.sha256(f"{self.seed}:retarget:{i}".encode("utf-8")).hexdigest()[:16], 16))
            new = rng.choice(unused or candidates)
            chosen[i] = {"old": old, "new": new, "prior_status": it.status}
            picked[(it.condition, it.split)].add(new)
        for i, c in chosen.items():
            it = self.items[i]
            it.meta["target_service"] = c["new"]
            it.tuple_labels["service_type"] = "unsupported"
            it.meta["is_unsupported"] = True
            it.status = "pending_gen"
            it.gen_batch_id = None
            it.ver_batch_id = None
            it.text = None
            it.generator = None
            it.verifier_labels = None
            it.verifier = None
            it.requeues.append({
                "op": "retarget", "retargeted_from": c["old"], "retargeted_to": c["new"],
                "prior_status": c["prior_status"], "reason": reason, "time": stamp,
            })
        self._save_state()
        return {"retargeted": chosen}

    def requeue_gen(self, batch_ids: list[str], reason: str = "requeue-gen") -> dict[str, Any]:
        """Withdraw gen todo batches that have no done file so their items are exported again.

        Items of the batches with status exported_gen return to pending_gen (the next export_gen builds
        their prompts from the current catalog) and each records {op, requeued_from, reason, archive, time}
        in `requeues`. The todo files move to gen/_superseded/<timestamp>/todo/ (never deleted) and the batch
        IDs are recorded as superseded: ingest_gen skips a late done file for them and export_gen never
        reuses them. Refused, with nothing changed: an unknown, superseded or ingested batch, a batch with a
        done file, and a batch holding items of a frozen (condition, split).
        """
        batch_ids = list(dict.fromkeys(batch_ids))
        if not batch_ids:
            raise ValueError("requeue_gen: no batch IDs given")
        members: dict[str, list[str]] = {}
        for bid in batch_ids:
            todo_file = self.gen_todo_dir / f"{bid}.jsonl"
            if bid in self.superseded_gen_batches:
                raise ValueError(f"requeue_gen: batch {bid} is already superseded")
            if not todo_file.exists():
                raise ValueError(f"requeue_gen: no gen todo batch {todo_file.name}")
            if (self.gen_done_dir / todo_file.name).exists():
                raise ValueError(f"requeue_gen: batch {bid} has a done file; ingest it instead")
            if todo_file.name in self.ingested_gen_files:
                raise ValueError(f"requeue_gen: batch {bid} was already ingested")
            with open(todo_file, "r", encoding="utf-8") as f:
                members[bid] = [json.loads(l)["item_id"] for l in f if l.strip()]
        reset = [
            i for bid, ids in members.items() for i in ids
            if i in self.items and self.items[i].gen_batch_id == bid and self.items[i].status == "exported_gen"
        ]
        self._refuse_frozen_changes("requeue_gen", reset)

        archive = self._archive_dir("gen")
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        requeued: list[str] = []
        skipped: list[str] = []
        for bid, ids in members.items():
            for i in ids:
                it = self.items.get(i)
                if it is None or it.gen_batch_id != bid or it.status != "exported_gen":
                    skipped.append(i)
                    continue
                it.status = "pending_gen"
                it.gen_batch_id = None
                it.requeues.append({
                    "op": "requeue_gen", "requeued_from": bid, "reason": reason, "archive": archive.name, "time": stamp,
                })
                requeued.append(i)
        self.superseded_gen_batches.update(batch_ids)
        self._save_state()

        # State first: a todo file left behind by an interruption belongs to a superseded batch and is ignored
        for bid in batch_ids:
            src = self.gen_todo_dir / f"{bid}.jsonl"
            if src.exists():
                (archive / "todo").mkdir(parents=True, exist_ok=True)
                src.replace(archive / "todo" / src.name)
        return {"requeued": requeued, "skipped": skipped, "archive_dir": str(archive)}

    def requeue_ver_items(self, item_ids: list[str], reason: str = "requeue-ver") -> dict[str, Any]:
        """Item-level requeue_ver: send these accepted/exported_ver items back to done_gen.

        Items whose ver batch is live go through requeue_ver (their batches' ver files are archived and the
        batches superseded, as the batch-level call does). Items whose batch is already superseded have no
        files left to move; they are reset the same way with an empty archive. Items with any other status
        are reported as skipped. Refused, with nothing changed: unknown IDs and items of a frozen
        (condition, split).
        """
        item_ids = list(dict.fromkeys(item_ids))
        if not item_ids:
            raise ValueError("requeue_ver_items: no item IDs given")
        unknown = [i for i in item_ids if i not in self.items]
        if unknown:
            raise ValueError(f"requeue_ver_items: unknown item IDs: {unknown}")
        eligible = [
            i for i in item_ids
            if self.items[i].status in ("exported_ver", "accepted") and self.items[i].ver_batch_id
        ]
        skipped = [i for i in item_ids if i not in eligible]
        self._refuse_frozen_changes("requeue_ver_items", eligible)

        live = [i for i in eligible if self.items[i].ver_batch_id not in self.superseded_ver_batches]
        dead = [i for i in eligible if i not in live]
        out: dict[str, Any] = {"requeued": [], "skipped": skipped, "archive_dir": None}
        if live:
            res = self.requeue_ver(sorted({self.items[i].ver_batch_id for i in live}), item_ids=live, reason=reason)
            out["requeued"].extend(res["requeued"])
            out["skipped"].extend(res["skipped"])
            out["archive_dir"] = res["archive_dir"]
        for i in dead:
            it = self.items[i]
            it.requeues.append({"requeued_from": it.ver_batch_id, "reason": reason, "archive": ""})
            it.status = "done_gen"
            it.ver_batch_id = None
            it.verifier_labels = None
            it.verifier = None
            out["requeued"].append(i)
        if dead:
            self._save_state()
        return out

    def compute_yields(self) -> dict[str, dict[str, Any]]:
        """First-pass and final yield per condition, from state.

        first_pass = original items accepted on their first attempt / target_n;
        final = accepted items (replacements included) / target_n.
        """
        yields: dict[str, dict[str, Any]] = {}
        for cond in sorted({it.condition for it in self.items.values()}):
            cond_items = [it for it in self.items.values() if it.condition == cond]
            target_n = sum(1 for it in cond_items if not it.is_replacement)
            accepted = [it for it in cond_items if it.status == "accepted"]
            first_pass = sum(1 for it in accepted if not it.is_replacement and it.attempt == 1)
            yields[cond] = {
                "target_n": target_n,
                "first_pass_accepted": first_pass,
                "accepted": len(accepted),
                "first_pass": first_pass / target_n if target_n else 0.0,
                "final": len(accepted) / target_n if target_n else 0.0,
            }
        return yields

    def assemble(
        self,
        conditions: list[str] | None = None,
        n_test: int = 300,
        n_dev: int = 60,
        splits: list[str] | None = None,
    ) -> dict[str, Any]:
        """Rewrite the formal corpus files from state (status == "accepted"), sorted by case_id.

        Writes data/<RQ>/<condition>/<split>.jsonl and _accepted/<condition>_<split>.jsonl for each selected
        (condition, split) - default: every group in state - after asserting n accepted items (n_test / n_dev)
        with unique case_ids for each; other groups' files are left untouched. With conditions or splits
        given, every group in their product is selected and each written group is recorded in _frozen.json.
        A frozen group must come out byte-identical. Nothing is written if any group fails. Also updates
        _yields.json for the selected conditions (entries of other conditions are kept).
        """
        selective = conditions is not None or splits is not None
        groups: dict[tuple[str, str], list[TargetItem]] = defaultdict(list)
        if conditions is not None:
            for cond in conditions:
                for sp in splits or ["test", "dev"]:
                    groups.setdefault((cond, sp), [])
        for it in self.items.values():
            if conditions is not None and it.condition not in conditions:
                continue
            if splits is not None and it.split not in splits:
                continue
            groups.setdefault((it.condition, it.split), [])
            if it.status == "accepted":
                groups[(it.condition, it.split)].append(it)

        for (cond, sp), accepted in sorted(groups.items()):
            n = n_test if sp == "test" else n_dev
            case_ids = [it.tuple_id for it in accepted]
            dupes = sorted(cid for cid, cnt in Counter(case_ids).items() if cnt > 1)
            if dupes:
                raise ValueError(f"assemble {cond}/{sp}: duplicate accepted case_ids {dupes}")
            if len(accepted) != n:
                raise ValueError(f"assemble {cond}/{sp}: {len(accepted)} accepted items, expected {n}")

        cat_dir = self.data_dir / "catalog"
        catalogs = load_catalogs(cat_dir)
        services_data = load_services(cat_dir)
        all_svcs = {
            s["id"]: s for s in services_data["services"] + services_data["novel_services"]
        }

        # Canonical path: data/edgebench/v1/<RQ>/<condition>/<split>.jsonl; frozen groups are checked first
        outputs: dict[Path, list[Case]] = {}
        for (cond, sp), accepted in sorted(groups.items()):
            accepted = sorted(accepted, key=lambda it: it.tuple_id)
            active_opts = self._get_active_service_options(cond, catalogs, all_svcs)
            canonical_file = self.data_dir / get_rq_for_condition(cond) / cond / f"{sp}.jsonl"
            outputs[canonical_file] = [it.to_case(active_service_options=active_opts) for it in accepted]
        blobs = check_frozen_outputs(self.data_dir, outputs)

        acc_dir = self.data_dir / "_accepted"
        acc_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for (cond, sp), accepted in sorted(groups.items()):
            accepted = sorted(accepted, key=lambda it: it.tuple_id)
            acc_file = acc_dir / f"{cond}_{sp}.jsonl"
            with open(acc_file, "w", encoding="utf-8") as f:
                for it in accepted:
                    record = {
                        "condition": it.condition,
                        "split": it.split,
                        "tuple_id": it.tuple_id,
                        "tuple_labels": it.tuple_labels,
                        "meta": it.meta,
                        "text": it.text,
                        "verifier_labels": it.verifier_labels,
                        "attempts": it.attempt,
                        "is_replacement": it.is_replacement,
                        "generator": it.generator,
                        "verifier": it.verifier,
                        "generator_model": it.generator,
                        "verifier_model": it.verifier,
                    }
                    f.write(json.dumps(record) + "\n")
            canonical_file = self.data_dir / get_rq_for_condition(cond) / cond / f"{sp}.jsonl"
            written.extend([acc_file, canonical_file])
        write_frozen_outputs(self.data_dir, blobs, freeze=selective)

        yields_file = self.data_dir / "_yields.json"
        yields: dict[str, dict[str, Any]] = {}
        if yields_file.exists():
            with open(yields_file, "r", encoding="utf-8") as f:
                yields = json.load(f)
        computed = self.compute_yields()
        for cond in sorted({c for c, _ in groups} & set(computed)):
            yields[cond] = computed[cond]
        with open(yields_file, "w", encoding="utf-8") as f:
            json.dump(yields, f, indent=2)

        return {"files": written, "yields": yields}

    def get_status_data(self) -> dict[str, Any]:
        """Aggregate status counts per condition."""
        status_data: dict[str, dict[str, Any]] = {}

        # Ensure all standard conditions are represented
        for cond in ALL_CONDITIONS:
            status_data[cond] = {
                "condition": cond,
                "rq": get_rq_for_condition(cond),
                "target_n": 0,
                "generated": 0,
                "verified": 0,
                "accepted": 0,
                "rejections": Counter(),
                "attempts": Counter(),
                "remaining": 0,
            }

        for it in self.items.values():
            cond = it.condition
            if cond not in status_data:
                status_data[cond] = {
                    "condition": cond,
                    "rq": get_rq_for_condition(cond),
                    "target_n": 0,
                    "generated": 0,
                    "verified": 0,
                    "accepted": 0,
                    "rejections": Counter(),
                    "attempts": Counter(),
                    "remaining": 0,
                }

            if not it.is_replacement:
                status_data[cond]["target_n"] += 1

            if it.text is not None:
                status_data[cond]["generated"] += 1

            if it.verifier_labels is not None or it.status == "accepted":
                status_data[cond]["verified"] += 1

            if it.status == "accepted":
                status_data[cond]["accepted"] += 1

            for r in it.reject_reasons:
                status_data[cond]["rejections"][r] += 1

            status_data[cond]["attempts"][f"att{it.attempt}"] += 1

        for cond, d in status_data.items():
            d["remaining"] = max(0, d["target_n"] - d["accepted"])

        return status_data

    def print_status(self) -> str:
        """Format and return a human-readable status table for all conditions."""
        data = self.get_status_data()

        headers = ["Condition", "Target N", "Generated", "Verified", "Accepted", "Rejected by Reason", "Attempts", "Remaining"]
        rows: list[list[str]] = []

        total_target = 0
        total_gen = 0
        total_ver = 0
        total_acc = 0
        total_rem = 0
        total_rejections: Counter[str] = Counter()
        total_attempts: Counter[str] = Counter()

        for cond in ALL_CONDITIONS:
            d = data.get(cond)
            if not d:
                continue

            target_n = d["target_n"]
            gen_n = d["generated"]
            ver_n = d["verified"]
            acc_n = d["accepted"]
            rem_n = d["remaining"]

            total_target += target_n
            total_gen += gen_n
            total_ver += ver_n
            total_acc += acc_n
            total_rem += rem_n
            total_rejections.update(d["rejections"])
            total_attempts.update(d["attempts"])

            rej_str = ", ".join(f"{k}: {v}" for k, v in d["rejections"].items()) if d["rejections"] else "none"
            att_str = ", ".join(f"{k}: {v}" for k, v in sorted(d["attempts"].items())) if d["attempts"] else "att1: 0"

            label = f"{d['rq']}/{cond}"
            rows.append([
                label,
                str(target_n),
                str(gen_n),
                str(ver_n),
                str(acc_n),
                rej_str,
                att_str,
                str(rem_n),
            ])

        total_rej_str = ", ".join(f"{k}: {v}" for k, v in total_rejections.items()) if total_rejections else "none"
        total_att_str = ", ".join(f"{k}: {v}" for k, v in sorted(total_attempts.items())) if total_attempts else "att1: 0"

        # Compute column widths
        widths = [len(h) for h in headers]
        for row in rows:
            for i, val in enumerate(row):
                widths[i] = max(widths[i], len(val))

        tot_row = [
            "TOTAL",
            str(total_target),
            str(total_gen),
            str(total_ver),
            str(total_acc),
            total_rej_str,
            total_att_str,
            str(total_rem),
        ]
        for i, val in enumerate(tot_row):
            widths[i] = max(widths[i], len(val))

        # Build table markdown
        header_line = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
        sep_line = "|-" + "-|-".join("-" * widths[i] for i in range(len(headers))) + "-|"
        body_lines = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |" for row in rows]
        total_line = "| " + " | ".join(tot_row[i].ljust(widths[i]) for i in range(len(headers))) + " |"

        table_md = "\n".join([header_line, sep_line] + body_lines + [sep_line, total_line])
        return table_md


def freeze_status(exchange_dir: str | Path, data_dir: str | Path) -> str:
    """Per (condition, split): accepted/target from exchange state and frozen yes/no from _frozen.json.

    Read-only: reads state.json and _frozen.json directly (constructing CorpusExchange creates directories).
    """
    state_file = Path(exchange_dir) / "_targets" / "state.json"
    items: list[TargetItem] = []
    if state_file.exists():
        with open(state_file, "r", encoding="utf-8") as f:
            items = [TargetItem.from_dict(d) for d in json.load(f).get("items", [])]
    frozen = load_frozen(data_dir)

    counts: dict[str, list[int]] = {}
    extra = sorted({it.condition for it in items} - set(ALL_CONDITIONS))
    for cond in ALL_CONDITIONS + extra:
        for sp in ("test", "dev"):
            counts[group_key(cond, sp)] = [0, 0]
    for it in items:
        c = counts[group_key(it.condition, it.split)]
        c[0] += it.status == "accepted"
        c[1] += not it.is_replacement

    rows = [["Group", "Accepted/Target", "Frozen"]]
    for key in list(counts) + sorted(set(frozen) - set(counts)):
        acc = f"{counts[key][0]}/{counts[key][1]}" if key in counts else "derived"
        rec = frozen.get(key)
        fz = f"yes {rec['sha256'][:12]} n={rec['n']}" if rec else "no"
        rows.append([key, acc, fz])
    widths = [max(len(r[i]) for r in rows) for i in range(3)]
    lines = ["| " + " | ".join(r[i].ljust(widths[i]) for i in range(3)) + " |" for r in rows]
    lines.insert(1, "|-" + "-|-".join("-" * w for w in widths) + "-|")
    return "\n".join(lines)
