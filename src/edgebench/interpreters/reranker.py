"""all-MiniLM-L6-v2 bi-encoder reranker reference interpreter for RQ4."""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from src.edgebench.contract import Case
from src.edgebench.interpreters.base import Decision, Interpreter

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PINNED_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "runs"
    / "EXP-2026-001"
    / "_reference"
    / "_models"
    / "reranker.json"
)


class RerankerInterpreter(Interpreter):
    platform: str | None = "M4-Max"

    def __init__(
        self,
        name: str = "MiniLM-Reranker",
        model_name_or_path: str = DEFAULT_MODEL_NAME,
        revision: str = PINNED_REVISION,
        threshold: float | None = None,
        config_path: str | Path | None = None,
        device: str | None = None,
        encoder: Any | None = None,
    ) -> None:
        super().__init__(name=name, deployment="reference")
        self.model_name_or_path = model_name_or_path
        self.revision = revision
        self.threshold = threshold
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self.device = device
        self._encoder = encoder
        self._desc_emb_cache: dict[str, Any] = {}

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

    def _ensure_encoder(self) -> Any:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            device = self._get_device()
            self._encoder = SentenceTransformer(
                self.model_name_or_path,
                revision=self.revision,
                device=device,
            )
        return self._encoder

    def _ensure_threshold(self) -> float:
        if self.threshold is not None:
            return self.threshold
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.threshold = float(data.get("theta", data.get("best_theta", 0.0)))
                return self.threshold
        # Fallback if config does not exist yet
        return 0.0

    def encode(self, texts: list[str]) -> Any:
        """Encode a list of strings into normalized embeddings."""
        encoder = self._ensure_encoder()
        return encoder.encode(texts, normalize_embeddings=True)

    def decide(self, case: Case) -> Decision:
        t_send_wall = time.time()
        t0 = time.perf_counter()

        theta = self._ensure_threshold()
        service_options = case.service_options or {}

        # Candidate services are active options excluding 'unsupported'
        candidates = [s for s in service_options.keys() if s != "unsupported"]

        if not candidates:
            predicted_service = "unsupported"
            max_cosine = 0.0
        else:
            query_emb = self.encode([case.text])
            if hasattr(query_emb, "cpu"):
                query_vec = query_emb.cpu().numpy()[0]
            else:
                query_vec = query_emb[0]

            cand_embs = []
            uncached_texts = []
            uncached_indices = []

            for idx, s in enumerate(candidates):
                desc = service_options[s]
                if desc in self._desc_emb_cache:
                    cand_embs.append(self._desc_emb_cache[desc])
                else:
                    cand_embs.append(None)
                    uncached_texts.append(desc)
                    uncached_indices.append(idx)

            if uncached_texts:
                new_embs = self.encode(uncached_texts)
                for u_idx, emb in zip(uncached_indices, new_embs):
                    cand_embs[u_idx] = emb
                    desc = service_options[candidates[u_idx]]
                    self._desc_emb_cache[desc] = emb

            import numpy as np

            cand_matrix = np.array(cand_embs)
            cosines = np.dot(cand_matrix, query_vec)

            best_idx = int(np.argmax(cosines))
            max_cosine = float(cosines[best_idx])

            if max_cosine < theta:
                predicted_service = "unsupported"
            else:
                predicted_service = candidates[best_idx]

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
            resolved_model="reference:reranker",
            provider="local",
            raw_response=f"reranker:{predicted_service}:{max_cosine:.4f}",
        )


def calibrate_reranker(
    dev_cases: list[Case],
    encoder: Any | None = None,
    observed_scores: list[tuple[float, str, str]] | None = None,
) -> tuple[float, float, list[dict[str, float]]]:
    """Calibrate theta on dev cases.

    Grid over observed cosines, pick the theta maximizing service top-1 accuracy (ties -> smaller theta).
    If observed_scores is provided, uses pre-computed (max_cosine, best_candidate, gold_service) tuples.
    Returns (best_theta, best_accuracy, accuracy_curve).
    """
    if observed_scores is None:
        if encoder is None:
            raise ValueError("Must provide either encoder or observed_scores")
        # Compute observed scores using encoder
        interpreter = RerankerInterpreter(encoder=encoder, threshold=0.0)
        scores_list = []
        for case in dev_cases:
            service_options = case.service_options or {}
            candidates = [s for s in service_options.keys() if s != "unsupported"]
            gold = case.truth[0].get("service_type", "unsupported") if case.truth else "unsupported"
            if not candidates:
                scores_list.append((-float("inf"), "unsupported", gold))
                continue
            query_emb = interpreter.encode([case.text])
            query_vec = query_emb.cpu().numpy()[0] if hasattr(query_emb, "cpu") else query_emb[0]
            cand_descs = [service_options[s] for s in candidates]
            cand_embs = interpreter.encode(cand_descs)
            import numpy as np

            cand_matrix = np.array(cand_embs)
            cosines = np.dot(cand_matrix, query_vec)
            best_idx = int(np.argmax(cosines))
            scores_list.append((float(cosines[best_idx]), candidates[best_idx], gold))
    else:
        scores_list = observed_scores

    # Collect observed candidate thetas
    cosines_only = [score[0] for score in scores_list if score[0] != -float("inf")]
    if not cosines_only:
        return 0.0, 0.0, [{"theta": 0.0, "accuracy": 0.0}]

    unique_cosines = sorted(list(set(cosines_only)))
    # Add boundary points: slightly below minimum observed and slightly above maximum
    min_c = unique_cosines[0]
    max_c = unique_cosines[-1]
    candidate_thetas = sorted(list(set([min_c - 1e-4] + unique_cosines + [max_c + 1e-4])))

    best_theta = candidate_thetas[0]
    best_acc = -1.0
    curve = []

    n = len(scores_list)
    for th in candidate_thetas:
        correct = 0
        for max_cos, best_cand, gold in scores_list:
            if max_cos < th:
                pred = "unsupported"
            else:
                pred = best_cand
            if pred == gold:
                correct += 1
        acc = correct / n if n > 0 else 0.0
        curve.append({"theta": round(float(th), 6), "accuracy": round(float(acc), 6)})

        if acc > best_acc:
            best_acc = acc
            best_theta = float(th)
        elif acc == best_acc:
            # Ties -> smaller theta
            if th < best_theta:
                best_theta = float(th)

    return best_theta, best_acc, curve
