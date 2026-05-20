# src/document_store.py
from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import Iterable, Dict, Any, List, Optional

class DocumentStore:
    def __init__(self, db_path: str = "data/document_store.sqlite3"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.init_schema()

    def init_schema(self):
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            file_name TEXT,
            path TEXT,
            status TEXT DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT,
            chunk_index INTEGER,
            text TEXT,
            status TEXT DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
        )
        """)
        self.conn.commit()

    def reset(self):
        cur = self.conn.cursor()
        cur.execute("DELETE FROM chunks")
        cur.execute("DELETE FROM documents")
        self.conn.commit()

    def upsert_document(self, doc_id: str, file_name: str, path: str, status: str = "active"):
        self.conn.execute(
            "INSERT OR REPLACE INTO documents(doc_id,file_name,path,status) VALUES(?,?,?,?)",
            (doc_id, file_name, path, status),
        )
        self.conn.commit()

    def insert_chunk(self, chunk_id: str, doc_id: str, chunk_index: int, text: str, status: str = "active"):
        self.conn.execute(
            "INSERT OR REPLACE INTO chunks(chunk_id,doc_id,chunk_index,text,status) VALUES(?,?,?,?,?)",
            (chunk_id, doc_id, chunk_index, text, status),
        )
        self.conn.commit()

    def active_documents(self) -> List[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM documents WHERE status='active' ORDER BY doc_id"))

    def close(self):
        self.conn.close()
