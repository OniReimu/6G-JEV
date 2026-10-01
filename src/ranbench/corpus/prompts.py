"""Generator prompt, blind-verifier context and interpreter context for RANIntent v1.

The verifier and every interpreter receive the same reading-rules artifact, loaded from
src/ranbench/schemas/ranintent_v1_reading_rules.md (load_reading_rules), never rebuilt per caller.
"""
from __future__ import annotations

import json
from typing import Any

from src.edgebench.corpus.tuples import TupleItem
from src.ranbench.corpus.spec import OPTIONS, PARAM_FIELDS, issuer_block
from src.ranbench.schemas import load_reading_rules

HUMAN_FIELD_NAMES: dict[str, str] = {
    "action": "What to do",
    "class": "Traffic class",
    "priority": "Priority level",
    "prb_share": "Radio resource share",
    "edge_site": "Edge processing location",
    "latency_target": "Delay target",
    "duration": "How long it applies",
}

CLUSTER_NAMES: dict[str, str] = {
    "north_cluster": "the north cluster",
    "south_cluster": "the south cluster",
    "stadium": "the stadium cluster",
    "hospital_zone": "the hospital zone",
}

STATE_LOCATION: dict[str, str] = {
    "most_loaded_cells": "the cells that are most heavily loaded (congested) right now",
    "worst_edge_cells": "the cells where users at the cell edge currently get the worst throughput",
}


def max_words_for(item: TupleItem, cfg: dict[str, Any]) -> int:
    t = cfg["text"]
    return t["max_words_long"] if item.meta.get("wording_family") in t["long_wording_families"] else t["max_words"]


def _issuer_line(issuer_id: str, cfg: dict[str, Any]) -> str:
    b = issuer_block(issuer_id, cfg)
    if b["kind"] == "operator":
        return f"{b['display_name']} (the operator; it may act on every traffic class)"
    owned = ", ".join(OPTIONS["class"][c] for c in b["owned_classes"])
    return (
        f"{b['display_name']}, a tenant that sends intents through the network exposure API and owns these traffic "
        f"classes: {owned}. Write as this tenant, about its own traffic"
    )


def build_generator_prompt(item: TupleItem, cfg: dict[str, Any]) -> str:
    """The full generator prompt for one tuple. It never contains target_cluster."""
    lab = item.tuple_labels
    specified = [f"{HUMAN_FIELD_NAMES['action']}: {OPTIONS['action'][lab['action']]}",
                 f"{HUMAN_FIELD_NAMES['class']}: {OPTIONS['class'][lab['class']]}"]
    unmentioned: list[str] = []
    for f in PARAM_FIELDS:
        if lab[f] == "unspecified":
            unmentioned.append(HUMAN_FIELD_NAMES[f])
        else:
            specified.append(f"{HUMAN_FIELD_NAMES[f]}: {OPTIONS[f][lab[f]]}")

    scope = lab["scope"]
    if scope in STATE_LOCATION:
        location = (
            f"Identify the cells only by their current state: {STATE_LOCATION[scope]}. The system finds these cells "
            "from live telemetry, so NEVER name or hint at any cluster, district, venue or landmark (no north, south, "
            "stadium, arena, hospital or any other place name), and do not guess which area it is."
        )
    elif scope in CLUSTER_NAMES:
        location = (
            f"Name the location explicitly as {CLUSTER_NAMES[scope]} (natural paraphrase allowed as long as the cluster "
            "is unmistakable); mention no other cluster or place."
        )
    elif scope == "all_cells":
        location = "The intent applies to the whole network; mention no cluster or place."
    else:
        location = "State no location at all."

    max_words = max_words_for(item, cfg)
    specified_str = "\n".join(f"  - {s}" for s in specified)
    unmentioned_str = "\n".join(f"  - {u}" for u in unmentioned) if unmentioned else "  - (none)"
    return (
        "You are an expert synthetic data generator writing realistic intents that a mobile network operator's RAN "
        "control system receives. Write a single natural intent text meeting the exact specifications below.\n\n"
        "HARD RULES:\n"
        "1. State every specified requirement exactly once in natural language.\n"
        "2. Do NOT state or imply any requirement for the fields to leave unmentioned.\n"
        "3. NEVER write internal identifiers or field names verbatim: nothing with an underscore (traffic class ids, "
        "field names such as target_cluster or prb_share, labels such as most_loaded_cells) and no telemetry column "
        "names.\n"
        f"4. Length: 1-3 sentences (email or formal SLA clause may be up to 4 sentences), at most {max_words} words.\n"
        "5. English only.\n"
        "6. NO VERBATIM PHRASE COPYING: never copy four consecutive words from the requirement descriptions below; "
        "express them in your own words.\n"
        "7. Do not invent telemetry values, cell identifiers or any requirement that is not listed.\n\n"
        "SPECIFICATIONS:\n"
        f"- Issuer: {_issuer_line(item.meta['issuer'], cfg)}.\n"
        f"- Wording family: {item.meta['wording_family']}\n"
        f"- Location: {location}\n"
        f"- Specified requirements:\n{specified_str}\n"
        f"- Fields to leave unmentioned:\n{unmentioned_str}\n\n"
        "Return only the intent text."
    )


def build_verifier_context(cfg: dict[str, Any]) -> str:
    """Blind-verifier instructions: the shared reading rules plus the batch I/O format; no targets, no conditions."""
    return (
        "# Blind verification instructions\n\n"
        "Each line of the batch file is one case: `item_id`, `issuer` (who sent the intent, the classes it owns and "
        "its rights), `text` (the intent) and `telemetry` (the KPM table the reading rules refer to). Label every case "
        "independently by applying the reading rules below; use only the case itself.\n\n"
        "Write one JSON line per case to the done file, in any order: "
        '`{"item_id": "<item_id>", "labels": {<the nine fields>}, "verifier": "<model id>"}`.\n\n'
        + load_reading_rules()
    )


def format_case(issuer: dict[str, Any], text: str, telemetry: str) -> str:
    """One case as an interpreter sees it: issuer block, intent text, rendered telemetry table."""
    return (
        "## Issuer\n" + json.dumps(issuer) + "\n\n"
        "## Intent\n" + text + "\n\n"
        "## Telemetry\n" + telemetry
    )


def build_interpreter_context(issuer: dict[str, Any], text: str, telemetry: str) -> dict[str, str]:
    """Interpreter input for adapters: {"rules": the shared reading-rules artifact, "case": format_case(...)}."""
    return {"rules": load_reading_rules(), "case": format_case(issuer, text, telemetry)}
