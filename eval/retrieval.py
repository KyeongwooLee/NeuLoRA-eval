from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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
        # Guard against duplicate IDs inside one build pass (Chroma rejects duplicates).
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
            # Chroma query `include` does not accept "ids" in recent versions.
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
    def __init__(self, router_model_path: Path, embedder: EmbeddingBackend):
        self.router_model_path = router_model_path
        self.embedder = embedder
        self.centroids: dict[str, list[float]] = {}
        self._load()

    def _load(self) -> None:
        if not self.router_model_path.exists():
            self.centroids = {}
            return
        payload = json.loads(self.router_model_path.read_text(encoding="utf-8"))
        raw_centroids = payload.get("centroids") or {}
        self.centroids = {
            str(style): [float(v) for v in vector]
            for style, vector in raw_centroids.items()
            if isinstance(vector, list)
        }

    def route(self, query: str, default_style: str = "direct") -> tuple[str, dict[str, float]]:
        if not self.centroids:
            return default_style, {default_style: 1.0}
        query_vector = self.embedder.encode_one(query)
        if not query_vector:
            return default_style, {default_style: 1.0}

        scores = {
            style: _cosine_similarity(query_vector, centroid)
            for style, centroid in self.centroids.items()
        }
        selected = max(scores, key=scores.get)
        return selected, scores
