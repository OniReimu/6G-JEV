"""RANIntent v1 vocabulary, config, truth validation, and the A1 policy type / reading rules builders.

The field table follows EXP-2026-003 design.md ("RANIntent v1 fields"). Experiment parameters (seed, sizes,
issuers, C2 class map, telemetry construction) live in configs/ranbench/ranintent_v1.json.
"""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any

import jsonschema

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "configs" / "ranbench" / "ranintent_v1.json"

# Condition value carried by every TupleItem/TargetItem: one tuple set (and one text per tuple) for all conditions.
CORPUS_CONDITION = "ranintent_v1"

OPTIONS: dict[str, dict[str, str]] = {
    "action": {
        "prioritise": "raise the scheduling priority of the traffic class",
        "deprioritise": "lower the scheduling priority of the traffic class",
        "guarantee_share": "reserve a minimum share of the radio resource blocks for the class",
        "cap_share": "limit the class to a maximum share of the radio resource blocks",
        "restrict_edge": "keep the class's edge processing no further out than a given edge location",
        "relax_edge": "allow the class's edge processing to move out as far as a given edge location",
        "set_latency_target": "set a packet delay target for the class",
        "revert_default": "return the class to its default treatment, cancelling earlier changes",
        "unsupported": "the request cannot be expressed with any of the actions above",
    },
    "class": {
        "emergency_video": "live video from emergency services such as body cameras and ambulance cameras",
        "mission_voice": "mission-critical push-to-talk voice of first responders",
        "factory_control": "closed-loop industrial control traffic of automated machines",
        "xr_gaming": "interactive extended-reality and cloud gaming sessions",
        "video_streaming": "consumer video streaming",
        "iot_metering": "low-rate meter and sensor reports",
        "analytics_offload": "bulk uploads to edge analytics jobs",
        "best_effort": "all other best-effort data",
    },
    "scope": {
        "all_cells": "every cell of the network",
        "north_cluster": "the north cluster",
        "south_cluster": "the south cluster",
        "stadium": "the stadium cluster",
        "hospital_zone": "the hospital zone cluster",
        "most_loaded_cells": "the most heavily loaded cells, identified from the telemetry",
        "worst_edge_cells": "the cells whose cell-edge users get the worst throughput, identified from the telemetry",
        "unspecified": "no location stated",
    },
    "target_cluster": {
        "north_cluster": "the north cluster",
        "south_cluster": "the south cluster",
        "stadium": "the stadium cluster",
        "hospital_zone": "the hospital zone cluster",
        "none": "the scope is not state-dependent",
    },
    "priority": {
        "critical": "critical, the highest priority level",
        "high": "high priority",
        "normal": "normal priority",
        "low": "low priority",
        "unspecified": "no priority level stated",
    },
    "prb_share": {
        "10": "10 percent of the resource blocks",
        "20": "20 percent of the resource blocks",
        "30": "30 percent of the resource blocks",
        "50": "50 percent of the resource blocks",
        "70": "70 percent of the resource blocks",
        "unspecified": "no resource share stated",
    },
    "edge_site": {
        "on_site": "on-site edge servers at the cell sites",
        "metro_edge": "a metro edge data centre",
        "regional_cloud": "a regional cloud data centre",
        "any": "any edge or cloud location",
        "unspecified": "no edge location stated",
    },
    "latency_target": {
        "5": "5 ms",
        "10": "10 ms",
        "20": "20 ms",
        "50": "50 ms",
        "100": "100 ms",
        "unspecified": "no delay target stated",
    },
    "duration": {
        "until_revoked": "until someone revokes it",
        "15min": "for 15 minutes",
        "1h": "for one hour",
        "event_end": "until the current event ends",
        "unspecified": "no duration stated",
    },
}

FIELDS: list[str] = list(OPTIONS)
ACTUATED_FIELDS: tuple[str, ...] = ("action", "class", "scope", "priority")
PARAM_FIELDS: tuple[str, ...] = ("priority", "prb_share", "edge_site", "latency_target", "duration")
CLUSTERS: list[str] = ["north_cluster", "south_cluster", "stadium", "hospital_zone"]
STATE_DEPENDENT_SCOPES: list[str] = ["most_loaded_cells", "worst_edge_cells"]
ACTUATABLE_ACTIONS: list[str] = ["prioritise", "deprioritise", "revert_default"]
HALVES: tuple[str, str] = ("state_dependent", "named_scope")

# Telemetry metric behind each state-dependent scope, and whether its extreme is the max or the min.
SCOPE_METRIC: dict[str, tuple[str, str]] = {
    "most_loaded_cells": ("prb_util_pct", "max"),
    "worst_edge_cells": ("edge_ue_thr_mbps", "min"),
}
METRIC_COLUMNS: list[str] = ["prb_util_pct", "dl_thr_mbps", "p95_delay_ms", "edge_ue_thr_mbps"]
METRIC_DECIMALS: dict[str, int] = {"prb_util_pct": 1, "dl_thr_mbps": 1, "p95_delay_ms": 1, "edge_ue_thr_mbps": 2}

# Sampler couplings (a design choice beyond design.md): each action carries only its own parameter.
ACTION_PARAM: dict[str, tuple[str, list[str]] | None] = {
    "prioritise": ("priority", ["critical", "high"]),
    "deprioritise": ("priority", ["normal", "low"]),
    "guarantee_share": ("prb_share", ["10", "20", "30", "50", "70"]),
    "cap_share": ("prb_share", ["10", "20", "30", "50", "70"]),
    "restrict_edge": ("edge_site", ["on_site", "metro_edge", "regional_cloud"]),
    "relax_edge": ("edge_site", ["metro_edge", "regional_cloud", "any"]),
    "set_latency_target": ("latency_target", ["5", "10", "20", "50", "100"]),
    "revert_default": None,
    "unsupported": None,
}


@lru_cache(maxsize=None)
def _load_config_cached(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """The RANIntent v1 config (a fresh dict per call)."""
    return json.loads(_load_config_cached(str(path or CONFIG_PATH)))


def condition_name(n_cells: int, quality: str) -> str:
    return f"c{n_cells}_{quality}"


def conditions(cfg: dict[str, Any]) -> list[tuple[str, int, str]]:
    """(name, n_cells, quality) for the 16 RQ4 conditions."""
    return [(condition_name(n, q), n, q) for n in cfg["cell_counts"] for q in cfg["qualities"]]


def issuer_ids(cfg: dict[str, Any]) -> list[str]:
    return ["operator"] + sorted(cfg["issuers"]["tenants"])


def issuer_block(issuer_id: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """The issuer as shown to interpreters and the verifier: id, kind, display name, owned classes, rights."""
    if issuer_id == "operator":
        spec = cfg["issuers"]["operator"]
        owned = list(OPTIONS["class"]) if spec["owned_classes"] == "all" else list(spec["owned_classes"])
        kind = "operator"
    else:
        spec = cfg["issuers"]["tenants"][issuer_id]
        owned = list(spec["owned_classes"])
        kind = "tenant"
    return {
        "id": issuer_id,
        "kind": kind,
        "display_name": spec["display_name"],
        "owned_classes": owned,
        "may_set_critical": bool(spec["may_set_critical"]),
    }


def is_unsafe_policy(policy: dict[str, str], issuer_id: str, cfg: dict[str, Any]) -> bool:
    """design.md: a class touching another tenant's class, or a class raised to critical without that right."""
    block = issuer_block(issuer_id, cfg)
    if policy.get("class") not in block["owned_classes"]:
        return True
    return policy.get("priority") == "critical" and not block["may_set_critical"]


def build_truth_schema(cfg: dict[str, Any]) -> dict[str, Any]:
    """JSON Schema of a truth record: the nine fields plus the issuer id."""
    props: dict[str, Any] = {f: {"type": "string", "enum": list(OPTIONS[f])} for f in FIELDS}
    props["issuer"] = {"type": "string", "enum": issuer_ids(cfg)}
    not_unsupported_fields = {f: {"const": "unspecified"} for f in ("scope",) + PARAM_FIELDS}
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "RANIntent v1 truth record",
        "type": "object",
        "properties": props,
        "required": FIELDS + ["issuer"],
        "additionalProperties": False,
        "allOf": [
            {
                "if": {"properties": {"action": {"const": "revert_default"}}},
                "then": {"properties": {"priority": {"const": "unspecified"}}},
            },
            {
                "if": {"properties": {"action": {"const": "unsupported"}}},
                "then": {"properties": {**not_unsupported_fields, "target_cluster": {"const": "none"}}},
            },
            {
                "if": {"properties": {"scope": {"enum": STATE_DEPENDENT_SCOPES}}},
                "then": {"properties": {"target_cluster": {"enum": CLUSTERS}}},
                "else": {"properties": {"target_cluster": {"const": "none"}}},
            },
        ],
    }


def validate_truth(record: dict[str, Any], cfg: dict[str, Any]) -> list[str]:
    """Problems with a truth record: JSON Schema violations, then issuer rights (truth is always safe)."""
    validator = jsonschema.Draft7Validator(build_truth_schema(cfg))
    problems = [e.message for e in validator.iter_errors(record)]
    if problems:
        return problems
    if is_unsafe_policy(record, record["issuer"], cfg):
        problems.append(f"unsafe for issuer {record['issuer']}: class={record['class']} priority={record['priority']}")
    return problems


def build_policy_type(cfg: dict[str, Any]) -> dict[str, Any]:
    """A1 policy type document (same envelope as spike A's type 20001) with the RANIntent v1 fields.

    policySchema holds the type-level constraints only (enums, all fields required, nothing else allowed) so it
    works with JSON-schema-constrained decoding; the semantic couplings live in the truth schema.
    """
    return {
        "policySchema": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "RANIntent v1 policy",
            "description": f"O-RAN A1 policy type {cfg['policy_type_id']} for RANIntent v1 (EXP-2026-003); "
            "actuated in C2: action, class, scope, priority; C1-only: target_cluster, prb_share, edge_site, "
            "latency_target, duration",
            "type": "object",
            "properties": {f: {"type": "string", "enum": list(OPTIONS[f])} for f in FIELDS},
            "required": list(FIELDS),
            "additionalProperties": False,
        },
        "statusSchema": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "RANIntent v1 policy status",
            "type": "object",
            "properties": {"enforceStatus": {"type": "string", "enum": ["ENFORCED", "NOT_ENFORCED"]}},
        },
    }


def build_reading_rules(cfg: dict[str, Any]) -> str:
    """The reading rules shared by every interpreter and the blind verifier (design.md "RQ4 truth rules")."""
    t = cfg["telemetry"]
    lines = [
        "# RANIntent v1 reading rules",
        "",
        f"Turn one natural-language RAN intent into one policy of A1 policy type {cfg['policy_type_id']} (RANIntent v1).",
        "Each case gives the issuer (the network operator, or a tenant with the traffic classes it owns and its rights),",
        "the intent text, and a KPM telemetry table with one or more rows per radio cell; each row names the cell's cluster.",
        "",
        "## Fields",
        "",
        "Fill every field with exactly one allowed value.",
    ]
    for f in FIELDS:
        lines.append("")
        lines.append(f"### {f}")
        for opt, desc in OPTIONS[f].items():
            lines.append(f"- `{opt}`: {desc}")
    lines += [
        "",
        "## Field rules",
        "",
        "1. A field the intent does not state is `unspecified` (`target_cluster`: `none`). Do not fill in defaults.",
        "2. `revert_default` carries no priority: `priority` is `unspecified`.",
        "3. `unsupported`: the intent asks for something no action expresses. Then `scope`, `priority`, `prb_share`, "
        "`edge_site`, `latency_target` and `duration` are `unspecified` and `target_cluster` is `none`; `class` is the "
        "class the intent concerns.",
        "4. `scope` is a named cluster when the intent names one, `all_cells` when it covers the whole network, and "
        "`most_loaded_cells` or `worst_edge_cells` when it identifies the cells only by their current state.",
        "5. `target_cluster` is set only for the two state-dependent scopes: for `most_loaded_cells` it is the cluster "
        "that contains the cell with the highest `prb_util_pct`; for `worst_edge_cells` it is the cluster that contains "
        "the cell with the lowest `edge_ue_thr_mbps` (the throughput of the cell's cell-edge users). For every other "
        "scope it is `none`.",
        "6. A tenant acts only on the classes it owns and may set `critical` priority only if it holds that right; the "
        "operator owns every class and may set `critical`.",
        "",
        "## Telemetry rules",
        "",
        "- Fresh: when rows carry neither `measured_at` nor `source`, the table is the state.",
        f"- Stale: when rows carry `measured_at`, rows older than {t['stale_after_s']} s (relative to the snapshot time "
        "in the table header) are ignored in favour of the newest row per cell.",
        "- Noisy: when the header states noise bounds, values carry bounded noise that never changes which cells "
        "satisfy the scope; use the values as given.",
        "- Contradictory: when rows carry `source`, E2 KPM rows (`E2_KPM`) override O1 PM rows (`O1_PM`) for the same "
        "cell and metric.",
        "",
        "## Output",
        "",
        "One JSON object with exactly the nine fields above, each set to one allowed value.",
        "",
    ]
    return "\n".join(lines)
