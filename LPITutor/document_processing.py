from __future__ import annotations

import os
from pathlib import Path

import chromadb
import pdfplumber
from sentence_transformers import SentenceTransformer


class DocumentProcessor:
    def __init__(
        self,
        pdf_dir: str,
        *,
        persist_directory: str = "./chroma_db",
        collection_name: str = "intelligent_tutor",
        embedding_model_name: str = "BAAI/bge-m3",
    ):
        self.pdf_dir = Path(pdf_dir)
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.embedder = SentenceTransformer(embedding_model_name)

        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))

    def _reset_collection(self) -> None:
        existing = [col.name for col in self.client.list_collections()]
        if self.collection_name in existing:
            self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(self.collection_name)

    @staticmethod
    def _read_pdf(pdf_file: Path) -> str:
        chunks: list[str] = []
        with pdfplumber.open(str(pdf_file)) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                text = text.strip()
                if text:
                    chunks.append(text)
        return "\n".join(chunks)

    @staticmethod
    def _split_text(text: str) -> list[str]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        merged: list[str] = []
        current = ""
        for line in lines:
            if len(current) + len(line) + 1 <= 350:
                current = f"{current} {line}".strip()
            else:
                if current:
                    merged.append(current)
                current = line
        if current:
            merged.append(current)
        return merged

    def upload_to_vector_db(self) -> int:
        self._reset_collection()
        if not self.pdf_dir.exists():
            return 0

        ids: list[str] = []
        docs: list[str] = []
        metas: list[dict] = []

        for pdf_file in sorted(self.pdf_dir.glob("*.pdf")):
            text = self._read_pdf(pdf_file)
            if not text:
                continue
            chunks = self._split_text(text)
            for idx, chunk in enumerate(chunks):
                ids.append(f"{pdf_file.stem}_{idx}")
                docs.append(chunk)
                metas.append({"source": pdf_file.name, "chunk_index": idx})

        if not docs:
            return 0

        embeddings = self.embedder.encode(docs, normalize_embeddings=True)
        self.collection.upsert(
            ids=ids,
            documents=docs,
            metadatas=metas,
            embeddings=[vector.tolist() for vector in embeddings],
        )
        return len(docs)


if __name__ == "__main__":
    processor = DocumentProcessor(pdf_dir="./pdf_files")
    count = processor.upload_to_vector_db()
    print(f"PDF processing and embedding upload completed. docs={count}")
