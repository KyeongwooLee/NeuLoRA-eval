from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from .common import ROUTER_MODE_CHOICES
except ImportError:  # pragma: no cover - script execution fallback
    from common import ROUTER_MODE_CHOICES
try:
    from router_utils import build_router_input
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from router_utils import build_router_input


@dataclass
class EmbeddingBackend:
    model_name: str

    def __post_init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]

    def encode_one(self, text: str) -> list[float]:
        vectors = self.encode([text])
        return vectors[0] if vectors else []


@dataclass
class MLPClassifierState:
    input_dim: int
    hidden_dim: int
    labels: list[str]
    fc1_weight: list[list[float]]
    fc1_bias: list[float]
    fc2_weight: list[list[float]]
    fc2_bias: list[float]


class ChromaCorpus:
    def __init__(self, persist_dir: Path, collection_name: str, embedder: EmbeddingBackend):
        import chromadb

        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedder = embedder
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))

        existing = [col.name for col in self.client.list_collections()]
        if self.collection_name in existing:
            self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(self.collection_name)

    def build(self, docs: Iterable[dict], batch_size: int = 128) -> int:
        docs = list(docs)
        if not docs:
            return 0

        total = 0
        id_counts: dict[str, int] = {}
        for start in range(0, len(docs), batch_size):
            batch = docs[start : start + batch_size]
            ids: list[str] = []
            texts: list[str] = []
            metas: list[dict] = []

            for item in batch:
                raw_id = str(item["id"])
                dup_idx = id_counts.get(raw_id, 0)
                safe_id = raw_id if dup_idx == 0 else f"{raw_id}__dup{dup_idx}"
                id_counts[raw_id] = dup_idx + 1

                meta = dict(item.get("metadata") or {})
                if dup_idx > 0:
                    meta.setdefault("original_id", raw_id)
                    meta["duplicate_index"] = dup_idx

                ids.append(safe_id)
                texts.append(str(item["text"]))
                metas.append(meta)

            vectors = self.embedder.encode(texts)
            self.collection.upsert(ids=ids, documents=texts, metadatas=metas, embeddings=vectors)
            total += len(batch)
        return total

    def query(self, query: str, top_k: int = 5) -> list[dict]:
        vector = self.embedder.encode_one(query)
        if not vector:
            return []
        result = self.collection.query(
            query_embeddings=[vector],
            n_results=max(1, top_k),
            include=["documents", "metadatas", "distances"],
        )
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        ids = result.get("ids", [[]])[0] if result.get("ids") else []

        rows: list[dict] = []
        for idx, doc in enumerate(docs):
            rows.append(
                {
                    "id": ids[idx] if idx < len(ids) else "",
                    "text": doc,
                    "metadata": metas[idx] if idx < len(metas) else {},
                    "distance": distances[idx] if idx < len(distances) else None,
                }
            )
        return rows


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return -1.0
    dim = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(dim))
    norm_a = math.sqrt(sum(a[i] * a[i] for i in range(dim)))
    norm_b = math.sqrt(sum(b[i] * b[i] for i in range(dim)))
    if norm_a <= 0 or norm_b <= 0:
        return -1.0
    return dot / (norm_a * norm_b)


class StyleRouter:
    def __init__(
        self,
        router_model_path: Path,
        embedder: EmbeddingBackend,
        *,
        mode: str = "cent",
        max_history_turns: int = 6,
        logger=None,
    ):
        self.router_model_path = router_model_path
        self.embedder = embedder
        self.mode = (mode or "cent").strip().lower()
        if self.mode not in ROUTER_MODE_CHOICES:
            raise ValueError(
                f"Unsupported router mode: {self.mode}. Supported: {', '.join(ROUTER_MODE_CHOICES)}"
            )
        self.max_history_turns = max(1, int(max_history_turns))
        self.logger = logger
        self.styles: list[str] = []
        self.centroids: dict[str, list[float]] = {}
        self.classifier: MLPClassifierState | None = None
        self._load()

    def _load(self) -> None:
        if not self.router_model_path.exists():
            if self.mode == "mlp":
                raise RuntimeError(f"Router model file not found for mlp mode: {self.router_model_path}")
            self.styles = []
            self.centroids = {}
            self.classifier = None
            return

        payload = json.loads(self.router_model_path.read_text(encoding="utf-8"))
        raw_styles = payload.get("styles") or []
        self.styles = [str(item).strip() for item in raw_styles if str(item).strip()]
        raw_centroids = payload.get("centroids") or {}
        self.centroids = {
            str(style): [float(v) for v in vector]
            for style, vector in raw_centroids.items()
            if isinstance(vector, list)
        }
        self.classifier = self._parse_classifier(payload.get("classifier"))
        if self.mode == "mlp" and self.classifier is None:
            raise RuntimeError(
                f"mlp router mode requires a valid classifier section in {self.router_model_path}"
            )

    def _parse_classifier(self, payload: dict | None) -> MLPClassifierState | None:
        if not isinstance(payload, dict):
            return None
        labels = payload.get("labels")
        state_dict = payload.get("state_dict")
        if not isinstance(labels, list) or not labels or not isinstance(state_dict, dict):
            return None
        try:
            fc1_weight = [[float(v) for v in row] for row in state_dict["fc1.weight"]]
            fc1_bias = [float(v) for v in state_dict["fc1.bias"]]
            fc2_weight = [[float(v) for v in row] for row in state_dict["fc2.weight"]]
            fc2_bias = [float(v) for v in state_dict["fc2.bias"]]
            input_dim = int(payload.get("input_dim") or len(fc1_weight[0]))
            hidden_dim = int(payload.get("hidden_dim") or len(fc1_weight))
        except Exception:
            return None

        if (
            not fc1_weight
            or not fc2_weight
            or len(fc1_weight) != len(fc1_bias)
            or len(fc2_weight) != len(fc2_bias)
            or len(fc2_weight) != len(labels)
        ):
            return None
        if any(len(row) != input_dim for row in fc1_weight):
            return None
        if any(len(row) != hidden_dim for row in fc2_weight):
            return None
        return MLPClassifierState(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            labels=[str(label) for label in labels],
            fc1_weight=fc1_weight,
            fc1_bias=fc1_bias,
            fc2_weight=fc2_weight,
            fc2_bias=fc2_bias,
        )

    @staticmethod
    def _softmax(scores: list[float]) -> list[float]:
        if not scores:
            return []
        max_score = max(scores)
        exps = [math.exp(score - max_score) for score in scores]
        total = sum(exps)
        if total <= 0:
            uniform = 1.0 / max(1, len(scores))
            return [uniform for _ in scores]
        return [value / total for value in exps]

    def _mlp_scores(self, query_vector: list[float]) -> dict[str, float]:
        classifier = self.classifier
        if classifier is None:
            raise RuntimeError("Router classifier is not loaded for mlp mode.")
        if len(query_vector) != classifier.input_dim:
            raise RuntimeError(
                f"Router input_dim mismatch: expected {classifier.input_dim}, got {len(query_vector)}"
            )

        hidden: list[float] = []
        for row, bias in zip(classifier.fc1_weight, classifier.fc1_bias):
            value = sum(weight * x for weight, x in zip(row, query_vector)) + bias
            hidden.append(max(0.0, value))

        logits: list[float] = []
        for row, bias in zip(classifier.fc2_weight, classifier.fc2_bias):
            value = sum(weight * x for weight, x in zip(row, hidden)) + bias
            logits.append(value)
        probs = self._softmax(logits)
        return {label: probs[idx] for idx, label in enumerate(classifier.labels)}

    def route(
        self,
        query: str,
        default_style: str = "direct",
        *,
        history: list[dict] | None = None,
        conversation_id: str | None = None,
    ) -> tuple[str, dict[str, float]]:
        query_text = build_router_input(
            current_query=query,
            history=history or [],
            conversation_id=str(conversation_id or ""),
            max_turns=self.max_history_turns,
            logger=self.logger,
        )

        if self.mode == "mlp":
            query_vector = self.embedder.encode_one(query_text)
            if not query_vector:
                raise RuntimeError("Router embedding returned an empty vector in mlp mode.")
            scores = self._mlp_scores(query_vector)
            if not scores:
                raise RuntimeError("Router classifier returned empty scores in mlp mode.")
            selected = max(scores, key=scores.get)
            return selected, scores

        if not self.centroids:
            return default_style, {default_style: 1.0}
        query_vector = self.embedder.encode_one(query_text)
        if not query_vector:
            return default_style, {default_style: 1.0}

        scores = {
            style: _cosine_similarity(query_vector, centroid)
            for style, centroid in self.centroids.items()
        }
        selected = max(scores, key=scores.get)
        return selected, scores
