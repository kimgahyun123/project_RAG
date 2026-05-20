# src/ingestion.py
from __future__ import annotations
import argparse
from pathlib import Path

try:
    from langchain_core.documents import Document
except Exception:
    from langchain.schema import Document

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.document_store import DocumentStore
from src.vector_store import VectorStoreManager

def infer_doc_id(path: Path) -> str:
    return path.stem

def load_raw_documents(raw_dir: str = "data/raw_docs"):
    raw_path = Path(raw_dir)
    files = sorted(raw_path.glob("*.txt"))
    docs = []
    for f in files:
        text = f.read_text(encoding="utf-8", errors="ignore")
        doc_id = infer_doc_id(f)
        docs.append((doc_id, f, text))
    return docs

def build_index(
    raw_dir: str = "data/raw_docs",
    db_path: str = "data/document_store.sqlite3",
    index_path: str = "data/faiss_index",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    reset: bool = True,
):
    store = DocumentStore(db_path)
    if reset:
        store.reset()

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    lc_docs = []

    for doc_id, f, text in load_raw_documents(raw_dir):
        store.upsert_document(doc_id=doc_id, file_name=f.name, path=str(f), status="active")
        chunks = splitter.split_text(text)
        for idx, chunk_text in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{idx:04d}"
            store.insert_chunk(chunk_id, doc_id, idx, chunk_text, status="active")
            lc_docs.append(Document(
                page_content=chunk_text,
                metadata={
                    "doc_id": doc_id,
                    "source_doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "chunk_index": idx,
                    "file_name": f.name,
                    "source": str(f),
                }
            ))

    vm = VectorStoreManager(index_path=index_path)
    vm.build(lc_docs)
    print(f"[ingestion] documents={len(load_raw_documents(raw_dir))}, chunks={len(lc_docs)}")
    print(f"[ingestion] sqlite={db_path}")
    print(f"[ingestion] faiss={index_path}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", default="data/raw_docs")
    p.add_argument("--db-path", default="data/document_store.sqlite3")
    p.add_argument("--index-path", default="data/faiss_index")
    p.add_argument("--chunk-size", type=int, default=1000)
    p.add_argument("--chunk-overlap", type=int, default=200)
    p.add_argument("--no-reset", action="store_true")
    args = p.parse_args()
    build_index(
        raw_dir=args.raw_dir,
        db_path=args.db_path,
        index_path=args.index_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        reset=not args.no_reset,
    )

if __name__ == "__main__":
    main()
