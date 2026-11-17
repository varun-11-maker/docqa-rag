"""Embeddings + persistent vector store (sentence-transformers + ChromaDB)."""
from collections.abc import Mapping
from typing import Any

import chromadb
import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer


class Store:
    def __init__(self, data_dir: str, model_name: str):
        self.model = SentenceTransformer(model_name)
        client = chromadb.PersistentClient(path=f"{data_dir}/chroma")
        self.col = client.get_or_create_collection(
            "documents", metadata={"hnsw:space": "cosine"}
        )

    def embed(self, texts: list[str]) -> NDArray[np.floating[Any]]:
        # Return a float32 ndarray — Chroma's typed API accepts NDArray embeddings
        # (list[list[float]] fails static checks due to list invariance).
        return np.asarray(
            self.model.encode(texts, normalize_embeddings=True), dtype=np.float32
        )

    def add(self, doc_id: str, name: str, chunks: list[dict]) -> None:
        """Add chunks shaped as {"text": str, "page": int | None}."""
        self.delete(doc_id)  # re-uploading the same file replaces it
        if not chunks:
            return
        self.col.add(
            ids=[f"{doc_id}:{i}" for i in range(len(chunks))],
            documents=[c["text"] for c in chunks],
            embeddings=self.embed([c["text"] for c in chunks]),
            metadatas=[
                {
                    "doc_id": doc_id,
                    "name": name,
                    "chunk": i,
                    # Chroma rejects None — use 0 for page-less text files
                    "page": int(c["page"]) if c.get("page") is not None else 0,
                }
                for i, c in enumerate(chunks)
            ],
        )

    def delete(self, doc_id: str) -> None:
        self.col.delete(where={"doc_id": doc_id})

    def list_docs(self) -> list[dict]:
        docs: dict[str, dict] = {}
        metas = self.col.get(include=["metadatas"]).get("metadatas") or []
        for m in metas:
            if not m:
                continue
            doc_id = str(m["doc_id"])
            d = docs.setdefault(
                doc_id, {"id": doc_id, "name": str(m["name"]), "chunks": 0}
            )
            d["chunks"] += 1
        return sorted(docs.values(), key=lambda d: d["name"])

    @staticmethod
    def _page_from_meta(meta: Mapping[str, Any]) -> int | None:
        page_val = meta.get("page", 0)
        if isinstance(page_val, bool) or page_val is None:
            page_int = 0
        elif isinstance(page_val, (int, float)):
            page_int = int(page_val)
        elif isinstance(page_val, str):
            try:
                page_int = int(page_val)
            except ValueError:
                page_int = 0
        else:
            page_int = 0
        return None if page_int == 0 else page_int

    def search(self, query: str, k: int) -> list[dict]:
        total = self.col.count()
        if total == 0:
            return []
        res = self.col.query(
            query_embeddings=self.embed([query]),
            n_results=min(k, total),
            include=["documents", "metadatas", "distances"],
        )
        documents = (res.get("documents") or [[]])[0] or []
        metadatas = (res.get("metadatas") or [[]])[0] or []
        distances = (res.get("distances") or [[]])[0] or []
        hits = []
        for text, meta, dist in zip(documents, metadatas, distances):
            if not meta:
                continue
            hits.append({
                "text": text,
                "doc": str(meta["name"]),
                "chunk": meta["chunk"],
                "page": self._page_from_meta(meta),
                "score": round(1 - float(dist), 3),
            })
        return hits
