# src/query_log.py
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_QUERY_LOG = Path("results/query_runs.jsonl")


def _doc_meta(item: Any) -> dict[str, Any]:
    doc = item[0] if isinstance(item, tuple) and len(item) >= 2 else item
    score = item[1] if isinstance(item, tuple) and len(item) >= 2 else None
    meta = getattr(doc, "metadata", {}) or {}
    text = getattr(doc, "page_content", "") or ""

    try:
        score = float(score) if score is not None else None
    except Exception:
        score = None

    return {
        "doc_id": str(meta.get("doc_id") or meta.get("source_doc_id") or "unknown"),
        "chunk_id": str(meta.get("chunk_id") or "unknown"),
        "chunk_index": meta.get("chunk_index", "NA"),
        "score": score,
        "text": text,
    }


def _unique(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def serialize_query_result(query: str, result: dict[str, Any]) -> dict[str, Any]:
    history = result.get("history", []) or []
    final_detection = result.get("detection")

    rounds = []
    run_quarantined_doc_ids = []

    for row in history:
        detection = row.get("detection")
        results = row.get("results", []) or []
        round_quarantine_doc_ids = list(getattr(detection, "quarantine_doc_ids", []) or [])
        run_quarantined_doc_ids.extend(round_quarantine_doc_ids)

        rounds.append({
            "round": row.get("round"),
            "label": getattr(detection, "label", ""),
            "score": getattr(detection, "risk_score", None),
            "via": getattr(detection, "via", ""),
            "rule_label": getattr(detection, "rule_label", ""),
            "rule_score": getattr(detection, "rule_score", None),
            "judge_label": getattr(detection, "judge_label", ""),
            "judge_score": getattr(detection, "judge_score", None),
            "reason": getattr(detection, "reason", ""),
            "judge_reason": getattr(detection, "judge_reason", ""),
            "quarantine_doc_ids": round_quarantine_doc_ids,
            "top_chunks": [_doc_meta(item) for item in results],
        })

    run_quarantined_doc_ids = result.get("run_quarantined_doc_ids") or _unique(run_quarantined_doc_ids)

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "query": query,
        "status": result.get("status", ""),
        "final_label": getattr(final_detection, "label", ""),
        "final_score": getattr(final_detection, "risk_score", None),
        "final_via": getattr(final_detection, "via", ""),
        "final_reason": getattr(final_detection, "reason", ""),
        "final_judge_reason": getattr(final_detection, "judge_reason", ""),
        "run_quarantined_doc_ids": run_quarantined_doc_ids,
        "quarantined_doc_ids": result.get("quarantined_doc_ids", []),
        "block_reason": result.get("block_reason", ""),
        "answer": result.get("answer", ""),
        "rounds": rounds,
    }


def append_query_run(query: str, result: dict[str, Any], path: str | Path = DEFAULT_QUERY_LOG) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    row = serialize_query_result(query=query, result=result)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
