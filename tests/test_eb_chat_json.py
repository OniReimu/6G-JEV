"""Tests for chat_json LLM adapter."""
from __future__ import annotations

import json
from unittest.mock import patch

from src.edgebench.contract import Case, FIELD_SETS
from src.edgebench.interpreters.chat_json import ChatJsonClient


def test_chat_json_schema_enums_k1():
    client = ChatJsonClient(
        name="DeepSeek-V4.1-Flash",
        model="deepseek/deepseek-v4.1-flash",
        provider_slug="together",
    )
    case = Case(
        case_id="c1",
        text="OCR text",
        fields=FIELD_SETS[4],
        bundle_size=1,
        truth=[{"service_type": "ocr", "locality": "unspecified", "quality_floor": "unspecified", "urgency": "unspecified"}],
    )
    schema = client.build_schema(case)
    assert schema["type"] == "object"
    assert "properties" in schema
    assert set(schema["required"]) == set(FIELD_SETS[4])
    for f in FIELD_SETS[4]:
        assert "enum" in schema["properties"][f]
        assert len(schema["properties"][f]["enum"]) > 0
    assert "count" in schema["properties"]["service_type"]["enum"]
    assert "unsupported" in schema["properties"]["service_type"]["enum"]


def test_chat_json_schema_k_greater_than_1():
    client = ChatJsonClient(
        name="GLM-5.3-Flash",
        model="z-ai/glm-5.3-flash",
        provider_slug="together",
    )
    case = Case(
        case_id="c2",
        text="1. Count objects. 2. Detect objects.",
        fields=FIELD_SETS[4],
        bundle_size=2,
        truth=[
            {"service_type": "count", "locality": "unspecified", "quality_floor": "unspecified", "urgency": "unspecified"},
            {"service_type": "detection", "locality": "unspecified", "quality_floor": "unspecified", "urgency": "unspecified"},
        ],
    )
    schema = client.build_schema(case)
    assert schema["type"] == "object"
    assert "requests" in schema["properties"]
    req_prop = schema["properties"]["requests"]
    assert req_prop["type"] == "array"
    assert req_prop["minItems"] == 2
    assert req_prop["maxItems"] == 2
    item_schema = req_prop["items"]
    assert item_schema["type"] == "object"
    assert set(item_schema["required"]) == set(FIELD_SETS[4])


def test_chat_json_reasoning_tokens_nonzero():
    client = ChatJsonClient(
        name="Qwen3.8-Flash",
        model="qwen/qwen3.8-flash",
        provider_slug="alibaba",
    )
    case = Case(
        case_id="c1",
        text="Count cars",
        fields=FIELD_SETS[4],
        bundle_size=1,
        truth=[{"service_type": "count", "locality": "unspecified", "quality_floor": "unspecified", "urgency": "unspecified"}],
    )
    mock_resp = {
        "model": "qwen/qwen3.8-flash",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps({"service_type": "count", "locality": "unspecified", "quality_floor": "unspecified", "urgency": "unspecified"})
                },
            }
        ],
        "usage": {
            "prompt_tokens": 120,
            "completion_tokens": 25,
            "completion_tokens_details": {"reasoning_tokens": 18},
            "cost": 0.0005,
        },
        "provider": "alibaba",
    }
    with patch("src.edgebench.interpreters.chat_json.make_request") as mock_make:
        mock_make.return_value = (200, json.dumps(mock_resp), 0.1, 1000.0, 1000.1, None, {})
        decision = client.decide(case)

    assert decision.valid is False
    assert decision.error_type == "reasoning_tokens_nonzero"
    assert decision.reasoning_tokens == 18


def test_chat_json_bad_json():
    client = ChatJsonClient(
        name="DeepSeek-V4.1-Flash",
        model="deepseek/deepseek-v4.1-flash",
        provider_slug="together",
    )
    case = Case(
        case_id="c1",
        text="Count cars",
        fields=FIELD_SETS[4],
        bundle_size=1,
        truth=[{"service_type": "count", "locality": "unspecified", "quality_floor": "unspecified", "urgency": "unspecified"}],
    )
    mock_resp = {
        "model": "deepseek/deepseek-v4.1-flash",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": "This is not valid JSON at all: {foo:"},
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 10, "cost": 0.0001},
        "provider": "together",
    }
    with patch("src.edgebench.interpreters.chat_json.make_request") as mock_make:
        mock_make.return_value = (200, json.dumps(mock_resp), 0.1, 1000.0, 1000.1, None, {})
        decision = client.decide(case)

    assert decision.valid is False
    assert decision.error_type == "bad_json"


def test_chat_json_reasoning_minimal_allowed():
    client = ChatJsonClient(
        name="GLM-5.3-Flash",
        model="z-ai/glm-5.3-flash",
        provider_slug="together",
        reasoning={"effort": "minimal"},
    )
    case = Case(
        case_id="c1",
        text="Count cars",
        fields=FIELD_SETS[4],
        bundle_size=1,
        truth=[{"service_type": "count", "locality": "unspecified", "quality_floor": "unspecified", "urgency": "unspecified"}],
    )
    req = client.build_request(case)
    # max_tokens = 512 + 24 * 4 * 1 = 608
    assert req["max_tokens"] == 608
    assert req["reasoning"] == {"effort": "minimal"}

    mock_resp = {
        "model": "z-ai/glm-5.3-flash",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps({"service_type": "count", "locality": "unspecified", "quality_floor": "unspecified", "urgency": "unspecified"})
                },
            }
        ],
        "usage": {
            "prompt_tokens": 120,
            "completion_tokens": 45,
            "completion_tokens_details": {"reasoning_tokens": 25},
            "cost": 0.0005,
        },
        "provider": "together",
    }
    with patch("src.edgebench.interpreters.chat_json.make_request") as mock_make:
        mock_make.return_value = (200, json.dumps(mock_resp), 0.1, 1000.0, 1000.1, None, {})
        decision = client.decide(case)

    assert decision.valid is True
    assert decision.error_type is None
    assert decision.reasoning_tokens == 25
    assert decision.output_tokens == 45
    assert decision.labels == [{"service_type": "count", "locality": "unspecified", "quality_floor": "unspecified", "urgency": "unspecified"}]

