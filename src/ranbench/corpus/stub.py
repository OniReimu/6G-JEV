"""Stub generator and stub blind verifier for dry runs of the RANIntent v1 exchange (no LLM calls).

The stub generator writes a templated text from fixed phrases; the stub verifier reads a ver batch line only
(text + telemetry), maps the phrases back to fields and resolves target_cluster with the reference reader. Items
they touch carry the generator/verifier ids below, which assembly refuses unless allow_stub is set.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from src.edgebench.corpus.tuples import TupleItem
from src.ranbench.corpus.spec import FIELDS
from src.ranbench.corpus.telemetry import parse_table_text, resolve_target_cluster

if TYPE_CHECKING:
    from src.ranbench.corpus.exchange import RanExchange

STUB_GENERATOR = "stub-generator"
STUB_VERIFIER = "stub-verifier"

STUB_PHRASES: dict[str, dict[str, str]] = {
    "action": {
        "prioritise": "boost",
        "deprioritise": "demote",
        "guarantee_share": "reserve a floor",
        "cap_share": "impose a ceiling",
        "restrict_edge": "pull processing inward",
        "relax_edge": "let processing spread outward",
        "set_latency_target": "pin a delay goal",
        "revert_default": "restore defaults",
        "unsupported": "do something unusual",
    },
    "class": {
        "emergency_video": "responder video",
        "mission_voice": "responder voice",
        "factory_control": "machine control",
        "xr_gaming": "immersive gaming",
        "video_streaming": "consumer streaming",
        "iot_metering": "meter reports",
        "analytics_offload": "analytics uploads",
        "best_effort": "background data",
    },
    "scope": {
        "all_cells": "across every cell",
        "north_cluster": "in the northern cluster",
        "south_cluster": "in the southern cluster",
        "stadium": "in the stadium cluster",
        "hospital_zone": "in the hospital zone",
        "most_loaded_cells": "where cells are busiest",
        "worst_edge_cells": "where edge users suffer most",
    },
    "priority": {"critical": "at critical level", "high": "at high level", "normal": "at normal level", "low": "at low level"},
    "prb_share": {
        "10": "ten percent of blocks",
        "20": "twenty percent of blocks",
        "30": "thirty percent of blocks",
        "50": "fifty percent of blocks",
        "70": "seventy percent of blocks",
    },
    "edge_site": {
        "on_site": "on cell-site servers",
        "metro_edge": "in a metro facility",
        "regional_cloud": "in a regional facility",
        "any": "anywhere at all",
    },
    "latency_target": {
        "5": "under five milliseconds",
        "10": "under ten milliseconds",
        "20": "under twenty milliseconds",
        "50": "under fifty milliseconds",
        "100": "under one hundred milliseconds",
    },
    "duration": {
        "until_revoked": "until told otherwise",
        "15min": "for a quarter hour",
        "1h": "for sixty minutes",
        "event_end": "until the event wraps up",
    },
}


def stub_text(item: TupleItem) -> str:
    lab = item.tuple_labels
    parts = [STUB_PHRASES[f][lab[f]] for f in STUB_PHRASES if lab[f] in STUB_PHRASES[f]]
    return f"Dry-run {item.split} intent {item.meta['item_index']:04d}: " + ", ".join(parts) + "."


def stub_labels(text: str, telemetry: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    for f in FIELDS:
        if f == "target_cluster":
            continue
        hits = [opt for opt, ph in STUB_PHRASES.get(f, {}).items() if re.search(rf"\b{re.escape(ph)}\b", text)]
        labels[f] = hits[0] if len(hits) == 1 else "unspecified"
    labels["target_cluster"] = resolve_target_cluster(parse_table_text(telemetry), labels["scope"])
    return {f: labels[f] for f in FIELDS}


def write_stub_gen_done(exchange: RanExchange) -> list[Path]:
    """Answer every open gen todo batch with stub texts."""
    written = []
    for todo in sorted(exchange.gen_todo_dir.glob("*.jsonl")):
        done = exchange.gen_done_dir / todo.name
        if done.exists() or todo.name in exchange.ingested_gen_files or todo.stem in exchange.superseded_gen_batches:
            continue
        with open(todo, encoding="utf-8") as fin, open(done, "w", encoding="utf-8") as fout:
            for line in fin:
                if line.strip():
                    iid = json.loads(line)["item_id"]
                    it = exchange.items[iid]
                    t_item = TupleItem(it.condition, it.split, it.tuple_id, it.tuple_labels, it.meta)
                    fout.write(json.dumps({"item_id": iid, "text": stub_text(t_item), "generator": STUB_GENERATOR}) + "\n")
        written.append(done)
    return written


def write_stub_ver_done(ver_todo_dir: Path, ver_done_dir: Path) -> list[Path]:
    """Answer every open ver todo batch from the batch lines alone (blind)."""
    written = []
    for todo in sorted(ver_todo_dir.glob("*.jsonl")):
        done = ver_done_dir / todo.name
        if done.exists():
            continue
        with open(todo, encoding="utf-8") as fin, open(done, "w", encoding="utf-8") as fout:
            for line in fin:
                if line.strip():
                    rec = json.loads(line)
                    labels = stub_labels(rec["text"], rec["telemetry"])
                    fout.write(json.dumps({"item_id": rec["item_id"], "labels": labels, "verifier": STUB_VERIFIER}) + "\n")
        written.append(done)
    return written
