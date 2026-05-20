# src/quarantine.py
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional


class QuarantineStore:
    def __init__(self, db_path: str = "data/quarantine.sqlite3"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.init_schema()

    def init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS quarantined_documents (
                doc_id TEXT PRIMARY KEY,
                label TEXT,
                status TEXT DEFAULT 'active',
                reason TEXT,
                query TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.commit()

    def quarantine_doc(
        self,
        doc_id: str,
        label: str = "malicious",
        reason: str = "",
        query: str = "",
        status: str = "active",
    ) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO quarantined_documents
            (doc_id, label, status, reason, query)
            VALUES (?, ?, ?, ?, ?)
            """,
            (doc_id, label, status, reason, query),
        )
        self.conn.commit()

    def active_doc_ids(self) -> set[str]:
        rows = self.conn.execute(
            "SELECT doc_id FROM quarantined_documents WHERE status='active'"
        ).fetchall()
        return {str(row["doc_id"]) for row in rows}

    def reset(self) -> None:
        self.conn.execute("DELETE FROM quarantined_documents")
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
