"""D-6: chat_json system prompt uses the shared READING_RULES exactly once."""
from __future__ import annotations

import pytest

from src.edgebench.contract import FIELD_SETS, READING_RULES, SERVICE_PRECEDENCE_RULE, Case
from src.edgebench.corpus.catalog import build_nested_and_churn_catalogs
from src.edgebench.corpus.catalog_data import RAW_SERVICES
from src.edgebench.interpreters.chat_json import ChatJsonClient

OLD_POLICY_PHRASES = ("Extract explicit requirements", "Respect negation")


def _unspec(fields: list[str]) -> dict[str, str]:
    return {f: "unspecified" for f in fields}


def _k15_options() -> dict[str, str]:
    descs = {s["id"]: s["description"] for s in RAW_SERVICES}
    opts = {sid: descs[sid] for sid in build_nested_and_churn_catalogs()["C_15"]}
    opts["unsupported"] = "unsupported service"
    return opts


def _single_case() -> Case:
    fields = FIELD_SETS[4]
    return Case(case_id="single", text="Read the plate.", fields=fields, truth=[_unspec(fields)])


def _bundle_case() -> Case:
    fields = FIELD_SETS[4]
    return Case(
        case_id="bundle",
        text="1. Count cars. 2. Read the plate.",
        fields=fields,
        bundle_size=2,
        truth=[_unspec(fields), _unspec(fields)],
    )


def _catalog_case() -> Case:
    fields = FIELD_SETS[4]
    return Case(
        case_id="catalog",
        text="Count cars.",
        fields=fields,
        service_options=_k15_options(),
        truth=[_unspec(fields)],
    )


CASES = {"single": _single_case, "bundle": _bundle_case, "catalog": _catalog_case}


@pytest.mark.parametrize("kind", sorted(CASES))
def test_d6_chat_json_system_prompt_reading_rules_once(kind):
    sys_prompt = ChatJsonClient(name="t", model="m", provider_slug="p").build_system_prompt(CASES[kind]())
    assert sys_prompt.count(READING_RULES) == 1
    assert sys_prompt.startswith(
        "Extract the requested fields from the requirements stated in the text. "
        + READING_RULES
        + " Return only compact JSON with the requested fields, using the semantic values below. No explanation.\n"
    )
    for phrase in OLD_POLICY_PHRASES:
        assert phrase not in sys_prompt
    expected_precedence = 1 if kind == "catalog" else 0
    assert sys_prompt.count(SERVICE_PRECEDENCE_RULE) == expected_precedence


def test_d6_chat_json_catalog_instruction_line_is_precedence_only():
    sys_prompt = ChatJsonClient(name="t", model="m", provider_slug="p").build_system_prompt(_catalog_case())
    instr = [l for l in sys_prompt.splitlines() if l.startswith("Instruction for ")]
    assert instr == [f"Instruction for service_type: {SERVICE_PRECEDENCE_RULE}"]
