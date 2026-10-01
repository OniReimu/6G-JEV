"""Prompt construction for the corpus generator and the blind verifier."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from src.edgebench.contract import DEFAULT_CRITERIA, get_field_instruction
from src.edgebench.corpus.tuples import TupleItem, is_rq4_condition

HUMAN_FIELD_NAMES: dict[str, str] = {
    "service_type": "Needed service capability",
    "locality": "Data locality / processing location",
    "quality_floor": "Processing quality tier",
    "urgency": "Urgency / priority",
    "retention": "Data retention / storage",
    "energy": "Energy profile / mode",
    "redundancy": "Instance redundancy / availability",
    "latency_class": "Latency deadline class",
}


DISTRACTOR_DEFINITION = (
    "a requirement value or service that the text explicitly marks as NOT wanted for this job, or that belongs to a "
    "different team, device or job (e.g. another crew's request, a setting used last month). The distractor must "
    "concern a field listed under \"Specified requirements\", must not change the requested value of that field, and "
    "must never mention a field listed under \"Fields to leave unmentioned\". Do not use ambient sensor readings or "
    "weather as the distractor."
)


def compute_prompt_hash(prompt_payload: dict[str, Any]) -> str:
    serialized = json.dumps(prompt_payload, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_generator_batch_prompt(
    items: list[TupleItem],
    service_descriptions: dict[str, str],
) -> tuple[str, str, dict[str, Any]]:
    """Build system prompt, user prompt, and json_schema for generator batch."""
    system_prompt = (
        "You are an expert synthetic data generator creating realistic edge service request texts. "
        "For each item in the batch, write a natural request text meeting the exact specifications.\n\n"
        "HARD RULES:\n"
        "1. State every specified requirement exactly once in natural language.\n"
        "2. Do NOT state or imply any requirement for unspecified fields.\n"
        "3. NEVER use field names (service_type, locality, quality_floor, urgency, retention, energy, redundancy, latency_class) "
        "or internal option IDs (site_only, remote_allowed, discard_after_use, standard, high, normal, urgent) verbatim.\n"
        "4. Length: 1–3 sentences (email or formal SLA clause may be up to 4 sentences).\n"
        "5. English unless code-switching instructions specify another language.\n"
        "6. Output must strictly conform to the provided JSON schema.\n"
        "7. FORBIDDEN SCAFFOLDS: Never use fixed scaffolds or repetitive framing. DO NOT use 'camera stream N' or 'camera N', "
        "'shift A/B' (or lettered shifts), 'Ticket #NNNN' or ticket headers, or 'Correction:' / 'CORRECTION:' prefix. "
        "Express operational context naturally without these scaffolds.\n"
        "8. NO VERBATIM PHRASE COPYING: Never copy contiguous 4-word spans from field criteria or service descriptions. "
        "Express requirements in your own words. Common 1-3 word terms like 'OCR', 'on site', 'urgent' alone are fine."
    )

    batch_specs: list[dict[str, Any]] = []
    for item in items:
        specified: list[str] = []
        unspecified: list[str] = []

        cond = item.condition
        norm_cond = cond.split("/")[-1]
        is_rq4 = is_rq4_condition(cond)

        for field_name, value in item.tuple_labels.items():
            human_field = HUMAN_FIELD_NAMES.get(field_name, field_name)
            if value == "unspecified":
                unspecified.append(human_field)
            else:
                if field_name == "service_type":
                    if is_rq4:
                        target_svc_id = item.meta.get("target_service", value)
                        desc = service_descriptions.get(
                            target_svc_id,
                            DEFAULT_CRITERIA["service_type"].get(target_svc_id, "required service"),
                        )
                    else:
                        desc = DEFAULT_CRITERIA["service_type"].get(value, value)
                    specified.append(f"{human_field}: {desc}")
                else:
                    desc = DEFAULT_CRITERIA[field_name].get(value, value)
                    specified.append(f"{human_field}: {desc}")

        wording = item.meta.get("wording_family", "casual chat")
        scenario = item.meta.get("scenario", "")

        # Condition specific instructions
        cond_inst = ""
        if cond in ("clean", "RQ2_clean"):
            cond_inst = "Write the request cleanly, directly, and unambiguously."
        elif "colloquial" in cond:
            cond_inst = "Use informal language, conversational tone, colloquial phrasing, slang, or casual fillers (e.g. 'hey', 'btw', 'kinda')."
        elif "negation" in cond:
            cond_inst = "Every specified requirement must be expressed through negation or exclusion where possible (e.g. 'don't let the photo leave this site', 'make sure you never send this offsite', 'there is no rush')."
        elif "codeswitch" in cond:
            lang = item.meta.get("language", "es")
            cond_inst = (
                f"Mix English and {lang} within the text: it must contain at least one full clause in English and at least one full clause in {lang}. "
                f"Do not write the whole text in one language. State each requirement once, in either language; never repeat a requirement as a translation."
            )
        elif "defaultbait" in cond:
            cond_inst = (
                "For 1 or 2 unspecified fields, include conversational context or hints that tempt a default assumption "
                "without actually stating a requirement (e.g., 'it is for tomorrow's report' tempts urgency but does not specify priority; "
                "'we might review this later' tempts retention but does not specify retention). Do not specify any requirement for those fields."
            )
        elif "revised" in cond:
            rev_field = item.meta.get("revision_field", "")
            rev_init_val = item.meta.get("revision_initial_value", "")
            if rev_field and rev_init_val:
                human_rev = HUMAN_FIELD_NAMES.get(rev_field, rev_field)
                if rev_field == "service_type":
                    init_desc = (
                        service_descriptions.get(rev_init_val, rev_init_val)
                        if is_rq4
                        else DEFAULT_CRITERIA["service_type"].get(rev_init_val, rev_init_val)
                    )
                    target_val = item.tuple_labels.get(rev_field, "")
                    target_desc = (
                        service_descriptions.get(target_val, target_val)
                        if is_rq4
                        else DEFAULT_CRITERIA["service_type"].get(target_val, target_val)
                    )
                else:
                    init_desc = DEFAULT_CRITERIA[rev_field].get(rev_init_val, rev_init_val)
                    target_val = item.tuple_labels.get(rev_field, "")
                    target_desc = DEFAULT_CRITERIA[rev_field].get(target_val, target_val)
                cond_inst = (
                    f"REVISION of {human_rev}: The text must first state an initial requirement of '{init_desc}', "
                    f"and later revise/update it to the final target '{target_desc}'. "
                    f"The earlier statement and the later update must both concern the revised field only; "
                    f"phrase the update the way a real operator would (e.g. a follow-up remark, a changed decision, a clarification from a colleague). "
                    f"Paraphrase both values; do not copy their wording, and do not mention any field listed under \"Fields to leave unmentioned\". "
                    f"Do NOT use 'Correction:' as a prefix. The verifier will extract the final target."
                )
            else:
                cond_inst = (
                    "At least one specified requirement must first be stated and later revised or corrected in the same request "
                    "(e.g., 'send it offsite... wait, actually keep all data strictly on premise'). Do NOT use 'Correction:' prefix."
                )
        elif "keyvalue" in cond:
            cond_inst = "Use a terse key-value or form style using natural words (e.g., 'task: read text | data: stays here | prio: high')."
        elif "low" in cond:
            cond_inst = "State each specified requirement directly and clearly once."
        elif "medium" in cond:
            cond_inst = (
                f"Include at least one specified requirement stated through negation, plus one distractor: {DISTRACTOR_DEFINITION}"
            )
        elif "high" in cond:
            rev_field = item.meta.get("revision_field", "")
            rev_init_val = item.meta.get("revision_initial_value", "")
            if rev_field and rev_init_val:
                human_rev = HUMAN_FIELD_NAMES.get(rev_field, rev_field)
                if rev_field == "service_type":
                    init_desc = (
                        service_descriptions.get(rev_init_val, rev_init_val)
                        if is_rq4
                        else DEFAULT_CRITERIA["service_type"].get(rev_init_val, rev_init_val)
                    )
                    target_val = item.tuple_labels.get(rev_field, "")
                    target_desc = (
                        service_descriptions.get(target_val, target_val)
                        if is_rq4
                        else DEFAULT_CRITERIA["service_type"].get(target_val, target_val)
                    )
                else:
                    init_desc = DEFAULT_CRITERIA[rev_field].get(rev_init_val, rev_init_val)
                    target_val = item.tuple_labels.get(rev_field, "")
                    target_desc = DEFAULT_CRITERIA[rev_field].get(target_val, target_val)
                revision_clause = (
                    f"(a) REVISION of {human_rev}: The text must first state an initial requirement of '{init_desc}', "
                    f"and later revise/update it to the final target '{target_desc}'. "
                    f"The earlier statement and the later update must both concern the revised field only; "
                    f"phrase the update the way a real operator would (e.g. a follow-up remark, a changed decision, a clarification from a colleague). "
                    f"Paraphrase both values; do not copy their wording, and do not mention any field listed under \"Fields to leave unmentioned\". "
                    f"Do NOT use 'Correction:' prefix."
                )
            else:
                revision_clause = "(a) At least one revision of a contract field."

            cond_inst = (
                f"Include at least one requirement stated through negation, plus one distractor: {DISTRACTOR_DEFINITION}\n"
                f"Also include:\n"
                f"{revision_clause}\n"
                f"(b) At least one requirement stated only indirectly (e.g. 'the footage must never leave the building' for site_only).\n"
                f"(c) One natural operational scope clause (e.g. 'covering the south docking bay during night operations'; do NOT use 'camera stream N' or 'shift A/B')."
            )
        elif is_rq4:
            target_svc = item.meta.get("target_service", item.tuple_labels.get("service_type"))
            is_unsupp = item.meta.get("is_unsupported", False)
            if is_unsupp:
                cond_inst = (
                    "The needed service is NOT supported by the catalog: describe the task accurately based on its specific function, "
                    "but ensure it does not sound like generic counting, generic object detection/localization, or text reading. "
                    "It must be classified as unsupported under the catalog precedence rule."
                )
            elif target_svc in ("count", "detection", "ocr"):
                cond_inst = (
                    f"The target service is generic ({target_svc}). The text must describe a general need for this capability "
                    f"and must NOT describe any specialised service's niche (no specific equipment or domain objects like license plates, drones, sea lice)."
                )
            else:
                cond_inst = (
                    f"The target service is specialised ({target_svc}). The text must describe a need that this specialised service "
                    f"fits better than any generic tool (e.g. mention specific domain objects, equipment, or operational context requiring this specialised capability)."
                )
        else:
            cond_inst = "State every specified requirement naturally."

        spec: dict[str, Any] = {"id": item.tuple_id}
        if is_rq4:
            spec["setting"] = (
                "choose one concrete, realistic deployment setting in which the needed capability is naturally used "
                "(vary the setting; do not default to a retail store, warehouse or factory unless the capability belongs there)."
            )
        else:
            spec["scenario"] = scenario
        spec["wording_family"] = wording
        spec["condition_instructions"] = cond_inst
        spec["specified_requirements"] = specified
        spec["fields_to_leave_unmentioned"] = unspecified
        batch_specs.append(spec)

    user_prompt = (
        "Generate a request text for each item in the following batch according to its specifications:\n\n"
        + json.dumps(batch_specs, indent=2)
    )

    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "text": {"type": "string"},
                    },
                    "required": ["id", "text"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }

    return system_prompt, user_prompt, schema


def build_verifier_batch_prompt(
    items: list[tuple[str, str]],  # (tuple_id, text)
    fields: list[str],
    service_options: dict[str, str] | None = None,
) -> tuple[str, str, dict[str, Any]]:
    """Build system prompt, user prompt, and json_schema for blind verifier batch."""
    system_prompt = (
        "You are an expert evaluator performing blind labeling of edge service requests. "
        "Label each text; for each field choose one allowed option following the instruction.\n\n"
        "FIELD CRITERIA:\n"
    )

    for field_name in fields:
        is_cat = field_name == "service_type" and service_options is not None
        if is_cat:
            opts = service_options
        else:
            opts = DEFAULT_CRITERIA[field_name]

        system_prompt += f"\n[{field_name}]\n"
        system_prompt += f"Instruction: {get_field_instruction(field_name, is_catalog=is_cat)}\n"
        for opt_val, opt_desc in opts.items():
            system_prompt += f"  - {opt_val}: {opt_desc}\n"

    batch_input = [{"id": tid, "text": txt} for tid, txt in items]
    user_prompt = (
        "Extract the exact specified requirements for each request text in the batch:\n\n"
        + json.dumps(batch_input, indent=2)
    )

    properties: dict[str, Any] = {}
    for f in fields:
        if f == "service_type":
            allowed_enums = list((service_options or DEFAULT_CRITERIA["service_type"]).keys())
        else:
            allowed_enums = list(DEFAULT_CRITERIA[f].keys())
        properties[f] = {"type": "string", "enum": allowed_enums}

    single_label_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "labels": {
                "type": "object",
                "properties": properties,
                "required": list(fields),
                "additionalProperties": False,
            },
        },
        "required": ["id", "labels"],
        "additionalProperties": False,
    }

    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": single_label_schema,
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }

    return system_prompt, user_prompt, schema


def build_single_item_generator_prompt(
    item: TupleItem,
    service_descriptions: dict[str, str],
) -> str:
    """Build full generator prompt text for a single exported item."""
    cond = item.condition
    norm_cond = cond.split("/")[-1]
    is_rq4 = is_rq4_condition(cond)

    specified: list[str] = []
    unspecified: list[str] = []

    for field_name, value in item.tuple_labels.items():
        human_field = HUMAN_FIELD_NAMES.get(field_name, field_name)
        if value == "unspecified":
            unspecified.append(human_field)
        else:
            if field_name == "service_type":
                if is_rq4:
                    target_svc_id = item.meta.get("target_service", value)
                    desc = service_descriptions.get(
                        target_svc_id,
                        DEFAULT_CRITERIA["service_type"].get(target_svc_id, "required service"),
                    )
                else:
                    desc = DEFAULT_CRITERIA["service_type"].get(value, value)
                specified.append(f"{human_field}: {desc}")
            else:
                desc = DEFAULT_CRITERIA[field_name].get(value, value)
                specified.append(f"{human_field}: {desc}")

    wording = item.meta.get("wording_family", "casual chat")
    scenario = item.meta.get("scenario", "")

    # Condition specific instructions
    cond_inst = ""
    if cond in ("clean", "RQ2_clean"):
        cond_inst = "Write the request cleanly, directly, and unambiguously."
    elif "colloquial" in cond:
        cond_inst = "Use informal language, conversational tone, colloquial phrasing, slang, or casual fillers (e.g. 'hey', 'btw', 'kinda')."
    elif "negation" in cond:
        cond_inst = "Every specified requirement must be expressed through negation or exclusion where possible (e.g. 'don't let the photo leave this site', 'make sure you never send this offsite', 'there is no rush')."
    elif "codeswitch" in cond:
        lang = item.meta.get("language", "es")
        cond_inst = (
            f"Mix English and {lang} within the text: it must contain at least one full clause in English and at least one full clause in {lang}. "
            f"Do not write the whole text in one language. State each requirement once, in either language; never repeat a requirement as a translation."
        )
    elif "defaultbait" in cond:
        cond_inst = (
            "For 1 or 2 unspecified fields, include conversational context or hints that tempt a default assumption "
            "without actually stating a requirement (e.g., 'it is for tomorrow's report' tempts urgency but does not specify priority; "
            "'we might review this later' tempts retention but does not specify retention). Do not specify any requirement for those fields."
        )
    elif "revised" in cond:
        rev_field = item.meta.get("revision_field", "")
        rev_init_val = item.meta.get("revision_initial_value", "")
        if rev_field and rev_init_val:
            human_rev = HUMAN_FIELD_NAMES.get(rev_field, rev_field)
            if rev_field == "service_type":
                init_desc = (
                    service_descriptions.get(rev_init_val, rev_init_val)
                    if is_rq4
                    else DEFAULT_CRITERIA["service_type"].get(rev_init_val, rev_init_val)
                )
                target_val = item.tuple_labels.get(rev_field, "")
                target_desc = (
                    service_descriptions.get(target_val, target_val)
                    if is_rq4
                    else DEFAULT_CRITERIA["service_type"].get(target_val, target_val)
                )
            else:
                init_desc = DEFAULT_CRITERIA[rev_field].get(rev_init_val, rev_init_val)
                target_val = item.tuple_labels.get(rev_field, "")
                target_desc = DEFAULT_CRITERIA[rev_field].get(target_val, target_val)
            cond_inst = (
                f"REVISION of {human_rev}: The text must first state an initial requirement of '{init_desc}', "
                f"and later revise/update it to the final target '{target_desc}'. "
                f"The earlier statement and the later update must both concern the revised field only; "
                f"phrase the update the way a real operator would (e.g. a follow-up remark, a changed decision, a clarification from a colleague). "
                f"Paraphrase both values; do not copy their wording, and do not mention any field listed under \"Fields to leave unmentioned\". "
                f"Do NOT use 'Correction:' as a prefix. The verifier will extract the final target."
            )
        else:
            cond_inst = (
                "At least one specified requirement must first be stated and later revised or corrected in the same request "
                "(e.g., 'send it offsite... wait, actually keep all data strictly on premise'). Do NOT use 'Correction:' prefix."
            )
    elif "keyvalue" in cond:
        cond_inst = "Use a terse key-value or form style using natural words (e.g., 'task: read text | data: stays here | prio: high')."
    elif "low" in cond:
        cond_inst = "State each specified requirement directly and clearly once."
    elif "medium" in cond:
        cond_inst = (
            f"Include at least one specified requirement stated through negation, plus one distractor: {DISTRACTOR_DEFINITION}"
        )
    elif "high" in cond:
        rev_field = item.meta.get("revision_field", "")
        rev_init_val = item.meta.get("revision_initial_value", "")
        if rev_field and rev_init_val:
            human_rev = HUMAN_FIELD_NAMES.get(rev_field, rev_field)
            if rev_field == "service_type":
                init_desc = (
                    service_descriptions.get(rev_init_val, rev_init_val)
                    if is_rq4
                    else DEFAULT_CRITERIA["service_type"].get(rev_init_val, rev_init_val)
                )
                target_val = item.tuple_labels.get(rev_field, "")
                target_desc = (
                    service_descriptions.get(target_val, target_val)
                    if is_rq4
                    else DEFAULT_CRITERIA["service_type"].get(target_val, target_val)
                )
            else:
                init_desc = DEFAULT_CRITERIA[rev_field].get(rev_init_val, rev_init_val)
                target_val = item.tuple_labels.get(rev_field, "")
                target_desc = DEFAULT_CRITERIA[rev_field].get(target_val, target_val)
            revision_clause = (
                f"(a) REVISION of {human_rev}: The text must first state an initial requirement of '{init_desc}', "
                f"and later revise/update it to the final target '{target_desc}'. "
                f"The earlier statement and the later update must both concern the revised field only; "
                f"phrase the update the way a real operator would (e.g. a follow-up remark, a changed decision, a clarification from a colleague). "
                f"Paraphrase both values; do not copy their wording, and do not mention any field listed under \"Fields to leave unmentioned\". "
                f"Do NOT use 'Correction:' prefix."
            )
        else:
            revision_clause = "(a) At least one revision of a contract field."

        cond_inst = (
            f"Include at least one requirement stated through negation, plus one distractor: {DISTRACTOR_DEFINITION}\n"
            f"Also include:\n"
            f"{revision_clause}\n"
            f"(b) At least one requirement stated only indirectly (e.g. 'the footage must never leave the building' for site_only).\n"
            f"(c) One natural operational scope clause (e.g. 'covering the south docking bay during night operations'; do NOT use 'camera stream N' or 'shift A/B')."
        )
    elif is_rq4:
        target_svc = item.meta.get("target_service", item.tuple_labels.get("service_type"))
        is_unsupp = item.meta.get("is_unsupported", False)
        if is_unsupp:
            cond_inst = (
                "The needed service is NOT supported by the catalog: describe the task accurately based on its specific function, "
                "but ensure it does not sound like generic counting, generic object detection/localization, or text reading. "
                "It must be classified as unsupported under the catalog precedence rule."
            )
        elif target_svc in ("count", "detection", "ocr"):
            cond_inst = (
                f"The target service is generic ({target_svc}). The text must describe a general need for this capability "
                f"and must NOT describe any specialised service's niche (no specific equipment or domain objects like license plates, drones, sea lice)."
            )
        else:
            cond_inst = (
                f"The target service is specialised ({target_svc}). The text must describe a need that this specialised service "
                f"fits better than any generic tool (e.g. mention specific domain objects, equipment, or operational context requiring this specialised capability)."
            )
    else:
        cond_inst = "State every specified requirement naturally."

    specified_str = "\n".join(f"- {s}" for s in specified) if specified else "- (none)"
    unspecified_str = "\n".join(f"- {u}" for u in unspecified) if unspecified else "- (none)"

    if is_rq4:
        scenario_line = (
            "- Setting: choose one concrete, realistic deployment setting in which the needed capability is naturally used "
            "(vary the setting; do not default to a retail store, warehouse or factory unless the capability belongs there).\n"
        )
    else:
        scenario_line = f"- Scenario: {scenario}\n"

    prompt = (
        "You are an expert synthetic data generator creating realistic edge service request texts. "
        "Write a single natural request text meeting the exact specifications below.\n\n"
        "HARD RULES:\n"
        "1. State every specified requirement exactly once in natural language.\n"
        "2. Do NOT state or imply any requirement for unspecified fields.\n"
        "3. NEVER use field names (service_type, locality, quality_floor, urgency, retention, energy, redundancy, latency_class) "
        "or internal option IDs (site_only, remote_allowed, discard_after_use, standard, high, normal, urgent) verbatim.\n"
        "4. Length: 1–3 sentences (email or formal SLA clause may be up to 4 sentences).\n"
        "5. English unless code-switching instructions specify another language.\n"
        "6. FORBIDDEN SCAFFOLDS: Never use fixed scaffolds or repetitive framing. DO NOT use 'camera stream N' or 'camera N', "
        "'shift A/B' (or lettered shifts), 'Ticket #NNNN' or ticket headers, or 'Correction:' / 'CORRECTION:' prefix. "
        "Express operational context naturally without these scaffolds.\n"
        "7. NO VERBATIM PHRASE COPYING: Never copy contiguous 4-word spans from field criteria or service descriptions. "
        "Express requirements in your own words. Common 1-3 word terms like 'OCR', 'on site', 'urgent' alone are fine.\n\n"
        f"SPECIFICATIONS:\n"
        f"{scenario_line}"
        f"- Wording family: {wording}\n"
        f"- Condition instructions: {cond_inst}\n"
        f"- Specified requirements:\n{specified_str}\n"
        f"- Fields to leave unmentioned:\n{unspecified_str}\n"
    )
    return prompt


def build_verifier_context_markdown(
    fields: list[str],
    service_options: dict[str, str] | None = None,
) -> str:
    """Build context markdown for blind verification.
    
    Contains verifier instructions, field definitions, and for RQ4 the active catalog
    (id + description + unsupported) with precedence rule.
    Contains NO target tuple, NO condition name that reveals a target, NO generator prompt.
    """
    lines = [
        "# Evaluator Instructions",
        "",
        "Label each text; for each field choose one allowed option following the instruction.",
        "",
        "## Field Criteria",
    ]

    for field_name in fields:
        is_cat = field_name == "service_type" and service_options is not None
        if is_cat:
            opts = service_options
        else:
            opts = DEFAULT_CRITERIA[field_name]

        lines.append(f"\n### {field_name}")
        instruction = get_field_instruction(field_name, is_catalog=is_cat)
        lines.append(f"Instruction: {instruction}\n")
        lines.append("Allowed Options:")
        for opt_val, opt_desc in opts.items():
            lines.append(f"  - `{opt_val}`: {opt_desc}")

    return "\n".join(lines) + "\n"
