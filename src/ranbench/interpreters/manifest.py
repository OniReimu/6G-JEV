"""Interpreter manifest for RANIntent v1 C1 (configs/ranbench/interpreters.json) and the adapter factory."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from src.edgebench.interpreters.base import Interpreter
from src.edgebench.manifest import PLATFORMS
from src.ranbench.interpreters.anyjev_l0 import AnyJevClient
from src.ranbench.interpreters.chat_json import RanChatJsonClient
from src.ranbench.interpreters.decisions import RanDecisionsClient

DEFAULT_MANIFEST_PATH = Path(__file__).resolve().parents[3] / "configs" / "ranbench" / "interpreters.json"


def load_manifest(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    with open(path or DEFAULT_MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)


def build_interpreter(
    name: str,
    manifest: dict[str, dict[str, Any]] | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Interpreter:
    """Adapter for a manifest entry. `base_url` overrides a self-hosted server URL (e.g. the per-job port)."""
    manifest = load_manifest() if manifest is None else manifest
    if name not in manifest:
        raise KeyError(f"Interpreter '{name}' not in the RANIntent manifest; available: {sorted(manifest)}")
    e = manifest[name]
    deployment = e["deployment"]
    if base_url is not None and deployment != "self-hosted":
        raise ValueError(f"base_url override is only allowed for self-hosted interpreters, not '{name}'")
    if e.get("platform") not in PLATFORMS:
        raise ValueError(f"'{name}' has unknown platform {e.get('platform')!r}; expected one of {PLATFORMS}")
    key = api_key if deployment == "hosted" else None
    adapter = e["adapter"]
    if adapter == "decisions":
        interp: Interpreter = RanDecisionsClient(
            name=name, model=e["model"], accepted_resolved_models=e.get("accepted_resolved_models"),
            base_url=base_url or e["base_url"], path=e["path"], api_key=key, deployment=deployment,
            timeout_s=e.get("timeout_s", 30.0),
        )
    elif adapter == "chat_json":
        interp = RanChatJsonClient(
            name=name, model=e["model"], provider_slug=e.get("provider"),
            accepted_resolved_models=e.get("accepted_resolved_models"), base_url=base_url or e["base_url"],
            path=e["path"], api_key=key, deployment=deployment, timeout_s=e.get("timeout_s", 30.0),
            reasoning=e.get("reasoning"),
        )
    elif adapter == "anyjev":
        # tokenizer: the local snapshot path (a shared HF cache on the cluster had no refs/main); ${HF_HOME} is expanded
        tokenizer = os.path.expandvars(e["tokenizer"]) if e.get("tokenizer") else None
        interp = AnyJevClient(
            name=name, backbone=e["backbone"], backend_url=base_url or e["backend_url"], tokenizer_name=tokenizer,
            level=e.get("level", "L0"), workers=e.get("workers", 16), timeout_s=e.get("timeout_s", 120.0),
            deployment=deployment,
        )
    else:
        raise ValueError(f"Unsupported adapter '{adapter}' for '{name}'")
    interp.platform = e["platform"]
    return interp
