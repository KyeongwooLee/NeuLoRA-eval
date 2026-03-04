from __future__ import annotations

import os

import chromadb
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer


class ResponseGenerator:
    def __init__(
        self,
        *,
        persist_directory: str = "./chroma_db",
        collection_name: str = "intelligent_tutor",
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        embedding_model_name: str = "BAAI/bge-m3",
        hf_api_key: str | None = None,
        hf_provider: str | None = None,
    ):
        self.model_name = model_name
        self.hf_api_key = hf_api_key or os.getenv("HF_API_KEY", "")
        self.hf_provider = (hf_provider or os.getenv("HF_PROVIDER") or "auto").strip().lower()
        if not self.hf_api_key:
            raise RuntimeError("HF_API_KEY is required.")

        self.client = InferenceClient(
            model=self.model_name,
            provider=self.hf_provider,
            token=self.hf_api_key,
        )
        chroma_client = chromadb.PersistentClient(path=persist_directory)
        self.collection = chroma_client.get_or_create_collection(collection_name)
        self.embedder = SentenceTransformer(embedding_model_name)

    def fetch_answer(self, query: str, top_k: int = 5) -> list[dict]:
        query_vector = self.embedder.encode([query], normalize_embeddings=True)[0].tolist()
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=max(1, top_k),
            include=["documents", "metadatas", "distances", "ids"],
        )
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        matched: list[dict] = []
        for i, text in enumerate(docs):
            matched.append(
                {
                    "text": text,
                    "metadata": metas[i] if i < len(metas) else {},
                    "distance": distances[i] if i < len(distances) else None,
                }
            )
        return matched

    def generate_response(self, query: str, matches: list[dict], level: str = "intermediate") -> str:
        context = "\n\n".join(item["text"] for item in matches[:5]).strip()
        user_prompt = (
            f"[Retrieved Context]\n{context}\n\n[Question]\n{query}\n\n"
            "Use the context when relevant."
        )
        system_prompt = (
            "You are a helpful and knowledgeable math tutor. "
            f"Explain at {level} level with clear reasoning and concise steps."
        )

        response = self.client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=384,
            temperature=0.2,
        )
        return (response.choices[0].message.content or "").strip()

    def respond_to_user(self, query: str, level: str = "intermediate") -> str:
        matches = self.fetch_answer(query)
        return self.generate_response(query=query, matches=matches, level=level)


if __name__ == "__main__":
    generator = ResponseGenerator()
    print(generator.respond_to_user("How do I multiply decimal numbers?", level="intermediate"))
