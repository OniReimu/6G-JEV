"""DistilBERT fine-tuned service classifier reference interpreter for RQ4."""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from src.edgebench.contract import Case
from src.edgebench.interpreters.base import Decision, Interpreter

DEFAULT_MODELS_ROOT = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "runs"
    / "EXP-2026-001"
    / "_reference"
    / "_models"
)


def mask_logits(
    logits: Any,
    id2label: dict[int, str],
    service_options: dict[str, str] | set[str] | list[str] | None,
) -> tuple[int, str, Any]:
    """Mask logits to active catalog options + 'unsupported', returning (best_idx, best_label, masked_logits)."""
    import numpy as np

    if service_options is None:
        allowed = {"unsupported"}
    elif isinstance(service_options, dict):
        allowed = set(service_options.keys())
    elif isinstance(service_options, (set, list)):
        allowed = set(service_options)
    else:
        allowed = set(service_options)

    allowed.add("unsupported")

    if hasattr(logits, "detach"):
        arr = logits.detach().cpu().numpy().copy()
    else:
        arr = np.array(logits, dtype=float).copy()

    for idx, label in id2label.items():
        if label not in allowed:
            arr[idx] = -float("inf")

    best_idx = int(np.argmax(arr))
    best_label = id2label[best_idx]
    return best_idx, best_label, arr


class ServiceClassifierInterpreter(Interpreter):
    platform: str | None = "M4-Max"

    def __init__(
        self,
        name: str = "DistilBERT-Clf-All",
        model_path: str | Path | None = None,
        heads: dict[str, str | Path] | None = None,
        device: str | None = None,
        condition: str | None = None,
        model: Any | None = None,
        tokenizer: Any | None = None,
        label2id: dict[str, int] | None = None,
        id2label: dict[int, str] | None = None,
        models_root: str | Path | None = None,
    ) -> None:
        super().__init__(name=name, deployment="reference")
        self.model_path = Path(model_path) if model_path else None
        self.device = device
        self.condition = condition
        self.models_root = Path(models_root) if models_root else DEFAULT_MODELS_ROOT
        if heads is not None:
            self.heads = {k: self._resolve_path(v) for k, v in heads.items()}
        elif "retrained" in self.name.lower():
            self.heads = {
                "churn25": self._resolve_path(self.models_root / "clf_retrained_25"),
                "churn50": self._resolve_path(self.models_root / "clf_retrained_50"),
            }
        else:
            self.heads = {}
        self._model = model
        self._tokenizer = tokenizer
        self._label2id = label2id
        self._id2label = id2label
        self._loaded_models: dict[str, tuple[Any, Any, dict[int, str]]] = {}

    def _resolve_path(self, p: str | Path) -> Path:
        path = Path(p)
        if path.is_absolute() and path.exists():
            return path
        if path.exists():
            return path
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        if (repo_root / path).exists():
            return repo_root / path
        if (self.models_root / path.name).exists():
            return self.models_root / path.name
        return path

    def _get_device(self) -> str:
        if self.device is not None:
            return self.device
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def resolve_head(self, condition: str | None = None) -> str:
        """Resolve head key ('_25' or '_50') from explicit mapping and condition."""
        cond = condition if condition is not None else self.condition
        if not cond:
            raise ValueError(
                f"No condition specified for retrained classifier '{self.name}'. "
                f"Available conditions: {list(self.heads.keys())}"
            )
        if cond not in self.heads:
            raise ValueError(
                f"Condition '{cond}' not supported for retrained classifier '{self.name}'. "
                f"Available conditions: {list(self.heads.keys())}"
            )
        head_path = str(self.heads[cond])
        if "_25" in head_path or cond == "churn25":
            return "_25"
        elif "_50" in head_path or cond == "churn50":
            return "_50"
        return f"_{cond}"

    def resolve_head_for_case(self, case: Case | None = None, condition: str | None = None) -> str:
        """Resolve head key based only on condition (explicit or from case.meta)."""
        cond = condition if condition is not None else self.condition
        if cond is None and case is not None and case.meta:
            cond = case.meta.get("condition")
        return self.resolve_head(cond)

    def _get_model_dir(self, head_key: str | None = None) -> Path:
        if "retrained" in self.name.lower() or (
            self.model_path is not None and "retrained" in str(self.model_path).lower()
        ):
            if head_key:
                if head_key in self.heads:
                    return self.heads[head_key]
                for k, v in self.heads.items():
                    if k.endswith(head_key.lstrip("_")) or head_key.lstrip("_") in k:
                        return v
            raise ValueError(f"No model directory mapped for retrained head '{head_key}'")

        if self.model_path is not None:
            resolved = self._resolve_path(self.model_path)
            if (resolved / "config.json").exists():
                return resolved
            return self.model_path

        # Derive from name
        name_lower = self.name.lower()
        if "all" in name_lower:
            return self.models_root / "clf_all"
        elif "frozen" in name_lower:
            return self.models_root / "clf_frozen"
        return self.models_root / "clf_all"

    def _ensure_loaded(self, head_key: str | None = None) -> tuple[Any, Any, dict[int, str]]:
        if self._model is not None and self._tokenizer is not None and self._id2label is not None:
            return self._model, self._tokenizer, self._id2label

        cache_key = head_key or "default"
        if cache_key in self._loaded_models:
            return self._loaded_models[cache_key]

        model_dir = self._get_model_dir(head_key)
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        device = self._get_device()
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        model.to(device)
        model.eval()

        id2label = {int(k): v for k, v in model.config.id2label.items()}
        self._loaded_models[cache_key] = (model, tokenizer, id2label)
        return model, tokenizer, id2label

    def decide(self, case: Case) -> Decision:
        t_send_wall = time.time()
        t0 = time.perf_counter()

        is_retrained = "retrained" in self.name.lower() or (
            self.model_path is not None and "retrained" in str(self.model_path).lower()
        )
        head_key = self.resolve_head_for_case(case) if is_retrained else None
        model, tokenizer, id2label = self._ensure_loaded(head_key)

        import torch

        device = self._get_device()
        with torch.no_grad():
            inputs = tokenizer(
                case.text,
                max_length=128,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            logits = outputs.logits[0]
            best_idx, predicted_service, _ = mask_logits(
                logits, id2label, case.service_options
            )

        labels = []
        for _ in range(case.bundle_size):
            req_labels: dict[str, str] = {}
            for f in case.fields:
                if f == "service_type":
                    req_labels[f] = predicted_service
                else:
                    req_labels[f] = "unspecified"
            labels.append(req_labels)

        latency_s = time.perf_counter() - t0
        t_recv_wall = time.time()

        if is_retrained:
            resolved_model = f"reference:clf_retrained{head_key}"
        elif "frozen" in self.name.lower():
            resolved_model = "reference:clf_frozen"
        else:
            resolved_model = "reference:clf_all"

        return Decision(
            labels=labels,
            valid=True,
            latency_s=latency_s,
            t_send_wall=t_send_wall,
            t_recv_wall=t_recv_wall,
            input_tokens=0,
            output_tokens=0,
            reasoning_tokens=0,
            cost_usd=0.0,
            resolved_model=resolved_model,
            provider="local",
            raw_response=f"classifier:{predicted_service}",
        )
