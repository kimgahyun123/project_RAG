# src/vector_store.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

try:
    from langchain_community.vectorstores import FAISS
except Exception as e:
    raise ImportError("langchain_community가 필요합니다: pip install -U langchain-community faiss-cpu") from e

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    from langchain_community.embeddings import HuggingFaceEmbeddings


def _embedding_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class VectorStoreManager:
    def __init__(
        self,
        index_path: str = "data/faiss_index",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
    ):
        self.index_path = Path(index_path)
        self.embedding_model = embedding_model
        self.device = device or _embedding_device()
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": self.device},
        )
        self.db = None

    def build(self, documents: Sequence[Any]):
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.db = FAISS.from_documents(list(documents), self.embeddings)
        self.db.save_local(str(self.index_path))
        return self.db

    def load(self):
        self.db = FAISS.load_local(
            str(self.index_path),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        return self.db

    def search(self, query: str, k: int = 5):
        if self.db is None:
            self.load()
        return self.db.similarity_search_with_score(query, k=k)
