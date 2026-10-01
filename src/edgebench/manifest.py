"""Manifest loader and factory for Edgebench interpreters."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from src.edgebench.interpreters.base import Interpreter
from src.edgebench.interpreters.chat_json import ChatJsonClient
from src.edgebench.interpreters.decisions import DecisionsClient
from src.edgebench.interpreters.oracle import OracleInterpreter
from src.edgebench.interpreters.rule import RuleInterpreter

DEFAULT_MANIFEST_PATH = (
    Path(__file__).resolve().parent.parent.parent / "configs" / "edgebench" / "models.json"
)
PLATFORMS = ("M4-Max", "H100-NVL", "hosted")


def load_manifest(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    manifest_path = Path(path) if path is not None else DEFAULT_MANIFEST_PATH
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_interpreter(
    display_name: str,
    manifest: dict[str, dict[str, Any]] | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    condition: str | None = None,
) -> Interpreter:
    """Build the interpreter for a manifest entry. `base_url` overrides the entry's URL (self-hosted only,
    e.g. the per-job port on the cluster); the effective URL is the interpreter's `base_url`."""
    if manifest is None:
        manifest = load_manifest()

    if display_name not in manifest:
        raise KeyError(
            f"Model '{display_name}' not found in edgebench manifest. Available: {list(manifest.keys())}"
        )

    entry = manifest[display_name]
    adapter = entry.get("adapter")
    deployment = entry.get("deployment", "hosted")
    if base_url is not None:
        if deployment != "self-hosted":
            raise ValueError(f"base_url override is only allowed for self-hosted models, not '{display_name}'")
        entry = dict(entry, base_url=base_url)
    platform = entry.get("platform")
    if platform is not None and platform not in PLATFORMS:
        raise ValueError(f"Model '{display_name}' has unknown platform '{platform}'; expected one of {PLATFORMS}")
    interpreter = _build(display_name, entry, adapter, deployment, api_key, condition=condition)
    interpreter.platform = platform
    if condition is not None and hasattr(interpreter, "condition") and not interpreter.condition:
        interpreter.condition = condition
    return interpreter


def _build(
    display_name: str,
    entry: dict[str, Any],
    adapter: str | None,
    deployment: str,
    api_key: str | None,
    condition: str | None = None,
) -> Interpreter:
    # API key from environment if not passed explicitly
    if api_key is None and deployment == "hosted":
        api_key = os.environ.get("OPENROUTER_API_KEY")

    if adapter == "decisions":
        return DecisionsClient(
            name=display_name,
            model=entry["model"],
            accepted_resolved_models=entry.get("accepted_resolved_models"),
            expected_provider=entry.get("provider"),
            base_url=entry.get("base_url", "https://openrouter.ai"),
            path=entry.get("path", "/api/alpha/decisions"),
            api_key=api_key,
            deployment=deployment,
            timeout_s=entry.get("timeout_s", 30.0),
        )
    elif adapter == "chat_json":
        return ChatJsonClient(
            name=display_name,
            model=entry["model"],
            provider_slug=entry.get("provider", "together"),
            accepted_resolved_models=entry.get("accepted_resolved_models"),
            base_url=entry.get("base_url", "https://openrouter.ai"),
            path=entry.get("path", "/api/v1/chat/completions"),
            api_key=api_key,
            deployment=deployment,
            timeout_s=entry.get("timeout_s", 30.0),
            reasoning=entry.get("reasoning"),
        )
    elif adapter == "rule":
        return RuleInterpreter(name=display_name)
    elif adapter == "oracle":
        return OracleInterpreter(name=display_name)
    elif adapter == "fixed-latency":
        from src.edgebench.interpreters.fixed_latency import FixedLatencyInterpreter
        return FixedLatencyInterpreter(name=display_name, delay_s=entry.get("delay_s", 0.6))
    elif adapter == "reranker":
        from src.edgebench.interpreters.reranker import RerankerInterpreter
        return RerankerInterpreter(
            name=display_name,
            model_name_or_path=entry.get("model", "sentence-transformers/all-MiniLM-L6-v2"),
            config_path=entry.get("config_path"),
            device=entry.get("device"),
        )
    elif adapter == "service_classifier":
        from src.edgebench.interpreters.service_classifier import ServiceClassifierInterpreter
        return ServiceClassifierInterpreter(
            name=display_name,
            model_path=entry.get("model"),
            heads=entry.get("heads"),
            condition=condition or entry.get("condition"),
            device=entry.get("device"),
        )
    else:
        raise ValueError(f"Unsupported adapter '{adapter}' for model '{display_name}'")
