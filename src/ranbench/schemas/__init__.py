"""Shared RANIntent v1 artifacts for the corpus and the interpreter adapters.

- ranintent_v1_policy_type.json: the A1 policy type (policySchema + statusSchema) registered in the A1 simulator;
  every interpreter output is validated against its policySchema.
- ranintent_v1_truth.schema.json: JSON Schema of a truth record (nine fields + issuer).
- ranintent_v1_reading_rules.md: the reading rules given to every interpreter and to the blind verifier.

The files are generated from src/ranbench/corpus/spec.py (`python -m src.ranbench.corpus.cli write-schemas`);
tests pin them to the generator.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema

SCHEMA_DIR = Path(__file__).resolve().parent
POLICY_TYPE_PATH = SCHEMA_DIR / "ranintent_v1_policy_type.json"
TRUTH_SCHEMA_PATH = SCHEMA_DIR / "ranintent_v1_truth.schema.json"
READING_RULES_PATH = SCHEMA_DIR / "ranintent_v1_reading_rules.md"


def load_policy_type() -> dict[str, Any]:
    return json.loads(POLICY_TYPE_PATH.read_text(encoding="utf-8"))


def load_policy_schema() -> dict[str, Any]:
    return load_policy_type()["policySchema"]


def load_truth_schema() -> dict[str, Any]:
    return json.loads(TRUTH_SCHEMA_PATH.read_text(encoding="utf-8"))


def load_reading_rules() -> str:
    return READING_RULES_PATH.read_text(encoding="utf-8")


def validate_policy(policy: Any) -> list[str]:
    """Schema-validity check of an interpreter output against the policy type; [] means valid."""
    validator = jsonschema.Draft7Validator(load_policy_schema())
    return [e.message for e in validator.iter_errors(policy)]
