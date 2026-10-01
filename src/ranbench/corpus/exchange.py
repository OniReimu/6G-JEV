"""File exchange for RANIntent v1 generation (Gemini-3.8-Flash) and blind verification (Claude-Opus-5.5), on edgebench's protocol.

Directory protocol (as edgebench):
  <exchange>/gen/todo/<batch>.jsonl   {item_id, split, prompt, max_words}       - exported here
  <exchange>/gen/done/<batch>.jsonl   {item_id, text, generator}                 - written by the generator
  <exchange>/ver/todo/<batch>.jsonl   {item_id, issuer, text, telemetry}         - blind: no truth, no tuple id
  <exchange>/ver/todo/<batch>.context.md  reading rules + I/O format
  <exchange>/ver/done/<batch>.jsonl   {item_id, labels, verifier}                - written by the verifier
  <exchange>/_targets/state.json      truth tuples and item state (never exposed)

One item per tuple (the text is shared by all 16 conditions). The verifier sees the text with the table of the
verify condition (config: 21 cells, fresh) and must reproduce all nine truth fields. State handling, retry (3
attempts) and replacement, requeue and done-file validation are inherited from edgebench's CorpusExchange.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from typing import Any

from src.edgebench.corpus.exchange import (
    CorpusExchange,
    TargetItem,
    _check_done_matches_todo,
    make_opaque_item_id,
)
from src.edgebench.corpus.lints import check_copy_4gram_lint
from src.edgebench.corpus.tuples import TupleItem
from src.ranbench.corpus.lints import (
    check_cluster_leak_lint,
    check_length_lint,
    check_literal_token_lint,
    diversity_violation,
    normalized_text,
)
from src.ranbench.corpus.prompts import build_generator_prompt, build_verifier_context, max_words_for
from src.ranbench.corpus.spec import CLUSTERS, CORPUS_CONDITION, FIELDS, OPTIONS, issuer_block, load_config
from src.ranbench.corpus.telemetry import build_table, render_table
from src.ranbench.corpus.tuples import sample_tuples
from src.ranbench.schemas import validate_policy

SPLITS = ("test", "dev")
LIVE_STATUSES = ("done_gen", "exported_ver", "accepted")
DESCRIPTIONS: list[str] = [d for opts in OPTIONS.values() for d in opts.values()]


def to_tuple_item(it: TargetItem) -> TupleItem:
    return TupleItem(it.condition, it.split, it.tuple_id, it.tuple_labels, it.meta)


def _read_done(done_file: Path, required: tuple[str, ...], stage: str) -> list[dict[str, Any]]:
    """Done-file validation as edgebench: JSON objects, required keys, no duplicate ids."""
    records, seen = [], set()
    with open(done_file, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise ValueError(f"Malformed {stage} done file {done_file.name} line {n}: invalid JSON: {e}")
            if not isinstance(obj, dict) or any(k not in obj for k in required):
                raise ValueError(f"Malformed {stage} done file {done_file.name} line {n}: needs keys {required}")
            iid = str(obj["item_id"]).strip()
            if iid in seen:
                raise ValueError(f"Malformed {stage} done file {done_file.name}: duplicate item_id '{iid}'")
            seen.add(iid)
            records.append(obj)
    if not records:
        raise ValueError(f"Malformed {stage} done file {done_file.name}: file is empty")
    # State changes (and the order-dependent diversity caps) never depend on line order in the file
    return sorted(records, key=lambda r: str(r["item_id"]).strip())


class RanExchange(CorpusExchange):
    def __init__(
        self,
        exchange_dir: str | Path = "runs/_ranintent_exchange",
        data_dir: str | Path = "data/ranbench/ranintent-v1",
        cfg: dict[str, Any] | None = None,
    ) -> None:
        self.cfg = cfg or load_config()
        super().__init__(exchange_dir=exchange_dir, data_dir=data_dir, seed=self.cfg["base_seed"])
        self.attempt_log = self.targets_dir / "attempt_log.jsonl"

    def _log_attempt(self, it: TargetItem, stage: str, outcome: str, agent: str) -> None:
        """Append-only record of every generation / verification outcome (per-generator acceptance)."""
        with open(self.attempt_log, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "item_id": it.item_id, "tuple_id": it.tuple_id, "split": it.split, "stage": stage,
                "generator": it.generator if stage == "ver" else agent, "agent": agent, "outcome": outcome,
            }) + "\n")

    def frozen_files(self) -> dict[str, str]:
        """{relative path: sha256} of the assembled corpus (manifest.json), {} before assembly."""
        path = self.data_dir / "manifest.json"
        return json.loads(path.read_text(encoding="utf-8"))["files"] if path.exists() else {}

    # A split is frozen once assemble has written <data_dir>/tuples/<split>.jsonl into the manifest
    def _in_frozen_group(self, it: TargetItem, frozen: dict[str, Any] | None = None) -> bool:
        return f"tuples/{it.split}.jsonl" in self.frozen_files()

    def init_targets(self, splits: list[str] | None = None, **_: Any) -> None:  # type: ignore[override]
        existing = {(it.split, it.tuple_id) for it in self.items.values()}
        added = False
        for sp in splits or list(SPLITS):
            for t in sample_tuples(sp, self.cfg):
                if (sp, t.tuple_id) in existing:
                    continue
                iid = make_opaque_item_id(t.tuple_id, CORPUS_CONDITION, sp)
                self.items[iid] = TargetItem(
                    item_id=iid, tuple_id=t.tuple_id, condition=CORPUS_CONDITION, split=sp,
                    tuple_labels=dict(t.tuple_labels), meta=dict(t.meta),
                )
                added = True
        if added:
            self._save_state()

    def _active_ids(self, dirs: tuple[Path, ...], ingested: set[str], superseded: set[str]) -> set[str]:
        ids: set[str] = set()
        for d in dirs:
            for f in d.glob("*.jsonl"):
                if f.name in ingested or f.stem in superseded:
                    continue
                with open(f, encoding="utf-8") as fh:
                    ids.update(json.loads(l)["item_id"] for l in fh if l.strip())
        return ids

    def export_gen(self, splits: list[str] | None = None, gen_batch_size: int = 100, **_: Any) -> list[Path]:  # type: ignore[override]
        """Gen todo batches for items that need a text; idempotent like edgebench's export_gen."""
        split_list = list(splits or SPLITS)
        self.init_targets(splits=split_list)
        active = self._active_ids((self.gen_todo_dir, self.gen_done_dir), self.ingested_gen_files, self.superseded_gen_batches)
        exported: list[Path] = []
        for sp in split_list:
            pending = [
                it for it in self.items.values()
                if it.split == sp and it.status in ("pending_gen", "exported_gen")
                and it.item_id not in active and not self._in_frozen_group(it)
            ]
            pending.sort(key=lambda it: it.item_id)
            for start in range(0, len(pending), gen_batch_size):
                n = 1
                while (self.gen_todo_dir / f"gen_{sp}_{n:03d}.jsonl").exists() or f"gen_{sp}_{n:03d}" in self.superseded_gen_batches:
                    n += 1
                batch_id = f"gen_{sp}_{n:03d}"
                path = self.gen_todo_dir / f"{batch_id}.jsonl"
                with open(path, "w", encoding="utf-8") as f:
                    for it in pending[start:start + gen_batch_size]:
                        prompt = build_generator_prompt(to_tuple_item(it), self.cfg)
                        it.meta["prompt_sha256"] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                        f.write(json.dumps({
                            "item_id": it.item_id, "split": sp, "prompt": prompt,
                            "max_words": max_words_for(to_tuple_item(it), self.cfg),
                        }) + "\n")
                        it.status, it.gen_batch_id = "exported_gen", batch_id
                exported.append(path)
        self._save_state()
        return exported

    def lint_text(self, it: TargetItem, text: str) -> str | None:
        """First failing static lint (reason string) or None."""
        t = self.cfg["text"]
        if not check_literal_token_lint(text)[0]:
            return "lint_literal_token"
        scope = it.tuple_labels["scope"]
        if not check_cluster_leak_lint(text, {scope} if scope in CLUSTERS else set())[0]:
            return "lint_cluster_leak"
        if not check_length_lint(text, t["min_words"], max_words_for(to_tuple_item(it), self.cfg))[0]:
            return "lint_length"
        if not check_copy_4gram_lint(text, DESCRIPTIONS)[0]:
            return "lint_copy_4gram"
        norm = normalized_text(text)
        for other in self.items.values():
            if other.tuple_id != it.tuple_id and other.status in LIVE_STATUSES and other.text and normalized_text(other.text) == norm:
                return "lint_duplicate"
        return None

    def ingest_gen(self) -> dict[str, Any]:  # type: ignore[override]
        results: dict[str, Any] = {"items_passed": 0, "items_rejected": 0, "rejections": Counter()}
        self._refuse_frozen_changes(
            "ingest_gen", self._pending_done_ids(self.gen_done_dir, self.ingested_gen_files, LIVE_STATUSES)
        )
        for done in sorted(self.gen_done_dir.glob("*.jsonl")):
            if done.name in self.ingested_gen_files or done.stem in self.superseded_gen_batches:
                continue
            records = _read_done(done, ("item_id", "text", "generator"), "gen")
            ids = {str(r["item_id"]).strip() for r in records}
            unknown = sorted(ids - set(self.items))
            if unknown:
                raise ValueError(f"Malformed gen done file {done.name}: unknown item_id(s) {unknown}")
            _check_done_matches_todo("gen", done, self.gen_todo_dir / done.name, ids)
            for rec in records:
                it = self.items[str(rec["item_id"]).strip()]
                if it.status in LIVE_STATUSES:
                    continue
                text = str(rec["text"]).strip()
                reason = self.lint_text(it, text)
                self._log_attempt(it, "gen", reason or "pass", str(rec["generator"]).strip())
                if reason:
                    self._reject_item(it, reason)
                    results["items_rejected"] += 1
                    results["rejections"][reason] += 1
                    continue
                it.text, it.generator, it.status = text, str(rec["generator"]).strip(), "done_gen"
                results["items_passed"] += 1
            self.ingested_gen_files.add(done.name)
            self._save_state()
        return results

    def verify_line(self, it: TargetItem) -> dict[str, Any]:
        """A blind ver line: issuer block, text and the verify-condition telemetry; nothing else."""
        vc = self.cfg["verify_condition"]
        table, _ = build_table(to_tuple_item(it), vc["cells"], vc["quality"], self.cfg)
        return {
            "item_id": it.item_id,
            "issuer": issuer_block(it.meta["issuer"], self.cfg),
            "text": it.text,
            "telemetry": render_table(table),
        }

    def export_ver(self, splits: list[str] | None = None, ver_batch_size: int = 100, **_: Any) -> list[Path]:  # type: ignore[override]
        split_list = list(splits or SPLITS)
        active = self._active_ids((self.ver_todo_dir, self.ver_done_dir), self.ingested_ver_files, self.superseded_ver_batches)
        pending = [
            it for it in self.items.values()
            if it.split in split_list and it.status in ("done_gen", "exported_ver") and it.text
            and it.item_id not in active and not self._in_frozen_group(it)
        ]
        if not pending:
            return []
        pending.sort(key=lambda it: it.item_id)
        random.Random(int(hashlib.sha256(f"{self.seed}:export_ver".encode()).hexdigest()[:16], 16)).shuffle(pending)
        context = build_verifier_context(self.cfg)
        exported: list[Path] = []
        for start in range(0, len(pending), ver_batch_size):
            n = 1
            while (
                (self.ver_todo_dir / f"vbatch_{n:04d}.jsonl").exists()
                or (self.ver_todo_dir / f"vbatch_{n:04d}.context.md").exists()
                or f"vbatch_{n:04d}" in self.superseded_ver_batches
            ):
                n += 1
            batch_id = f"vbatch_{n:04d}"
            jsonl = self.ver_todo_dir / f"{batch_id}.jsonl"
            with open(jsonl, "w", encoding="utf-8") as f:
                for it in pending[start:start + ver_batch_size]:
                    f.write(json.dumps(self.verify_line(it)) + "\n")
                    it.status, it.ver_batch_id = "exported_ver", batch_id
            ctx = self.ver_todo_dir / f"{batch_id}.context.md"
            ctx.write_text(context, encoding="utf-8")
            exported += [jsonl, ctx]
        self._save_state()
        return exported

    def ingest_ver(self, check_diversity: bool = True) -> dict[str, Any]:  # type: ignore[override]
        """Accept an item when the verifier reproduces all nine fields and (optionally) diversity caps hold."""
        t = self.cfg["text"]
        results: dict[str, Any] = {
            "items_accepted": 0, "items_rejected": 0, "rejections": Counter(), "mismatched_fields": Counter(),
        }
        self._refuse_frozen_changes(
            "ingest_ver", self._pending_done_ids(self.ver_done_dir, self.ingested_ver_files, ("accepted",))
        )
        for done in sorted(self.ver_done_dir.glob("*.jsonl")):
            if done.name in self.ingested_ver_files or done.stem in self.superseded_ver_batches:
                continue
            records = _read_done(done, ("item_id", "labels", "verifier"), "ver")
            ids = {str(r["item_id"]).strip() for r in records}
            unknown = sorted(ids - set(self.items))
            if unknown:
                raise ValueError(f"Malformed ver done file {done.name}: unknown item_id(s) {unknown}")
            if any(not isinstance(r["labels"], dict) for r in records):
                raise ValueError(f"Malformed ver done file {done.name}: 'labels' must be a dict")
            _check_done_matches_todo("ver", done, self.ver_todo_dir / done.name, ids)
            for rec in records:
                it = self.items[str(rec["item_id"]).strip()]
                if it.status == "accepted":
                    continue
                labels = rec["labels"]
                verifier = str(rec["verifier"]).strip()
                # Labels must be a valid policy: exactly the nine fields, each an allowed value
                if set(labels) != set(FIELDS) or validate_policy(labels):
                    self._log_attempt(it, "ver", "verifier_invalid_labels", verifier)
                    self._reject_item(it, "verifier_invalid_labels")
                    results["items_rejected"] += 1
                    results["rejections"]["verifier_invalid_labels"] += 1
                    continue
                wrong = [f for f in FIELDS if labels[f] != it.tuple_labels[f]]
                if wrong:
                    results["mismatched_fields"].update(wrong)
                    self._log_attempt(it, "ver", "verifier_mismatch:" + ",".join(wrong), verifier)
                    self._reject_item(it, "verifier_mismatch")
                    results["items_rejected"] += 1
                    results["rejections"]["verifier_mismatch"] += 1
                    continue
                if check_diversity:
                    accepted = [x.text or "" for x in self.items.values() if x.split == it.split and x.status == "accepted"]
                    total = sum(1 for x in self.items.values() if x.split == it.split and not x.is_replacement)
                    reason = diversity_violation(it.text or "", accepted, total, t["max_opening_4gram_share"], t["max_5gram_share"])
                    if reason:
                        self._log_attempt(it, "ver", reason, verifier)
                        self._reject_item(it, reason)
                        results["items_rejected"] += 1
                        results["rejections"][reason] += 1
                        continue
                self._log_attempt(it, "ver", "accepted", verifier)
                it.status, it.verifier_labels, it.verifier = "accepted", dict(labels), verifier
                results["items_accepted"] += 1
            self.ingested_ver_files.add(done.name)
            self._save_state()
        return results

    def yields(self) -> dict[str, Any]:
        """First-pass / final yield per split and acceptance per generator (EXP-2026-001 D-4 control)."""
        out: dict[str, Any] = {}
        for sp in sorted({it.split for it in self.items.values()}):
            items = [it for it in self.items.values() if it.split == sp]
            target = sum(1 for it in items if not it.is_replacement)
            acc = [it for it in items if it.status == "accepted"]
            first = sum(1 for it in acc if not it.is_replacement and it.attempt == 1)
            out[sp] = {
                "target_n": target, "accepted": len(acc), "first_pass_accepted": first,
                "first_pass": first / target if target else 0.0, "final": len(acc) / target if target else 0.0,
            }
        gens: dict[str, Counter[str]] = {}
        if self.attempt_log.exists():
            for line in self.attempt_log.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                c = gens.setdefault(rec["generator"], Counter())
                if rec["stage"] == "gen":
                    c["texts"] += 1
                    c["passed_lints"] += rec["outcome"] == "pass"
                else:
                    c["verified"] += 1
                    c["accepted"] += rec["outcome"] == "accepted"
        out["per_generator"] = {
            g: dict(c) | {"acceptance": c["accepted"] / c["texts"] if c["texts"] else 0.0} for g, c in sorted(gens.items())
        }
        return out

    def print_status(self) -> str:  # type: ignore[override]
        rows = ["| Split | Target | Pending gen | Exported gen | Done gen | Exported ver | Accepted | Rejections |", "|---|---|---|---|---|---|---|---|"]
        for sp in SPLITS:
            items = [it for it in self.items.values() if it.split == sp]
            st = Counter(it.status for it in items)
            rej = Counter(r for it in items for r in it.reject_reasons)
            rows.append(
                f"| {sp} | {sum(not it.is_replacement for it in items)} | {st['pending_gen']} | {st['exported_gen']} | "
                f"{st['done_gen']} | {st['exported_ver']} | {st['accepted']} | "
                + (", ".join(f"{k}: {v}" for k, v in sorted(rej.items())) or "none") + " |"
            )
        return "\n".join(rows)
