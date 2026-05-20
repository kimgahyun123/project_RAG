# src/context_builder.py
from __future__ import annotations
from typing import Any, Sequence

def _unpack(item: Any):
    if isinstance(item, tuple) and len(item) >= 2:
        return item[0], item[1]
    return item, None

def build_context(results: Sequence[Any], include_metadata: bool = True, max_chars: int | None = None) -> str:
    parts = []
    for rank, item in enumerate(results, start=1):
        doc, score = _unpack(item)
        metadata = getattr(doc, "metadata", {}) or {}
        text = getattr(doc, "page_content", "") or ""
        if include_metadata:
            score_txt = f"{float(score):.4f}" if score is not None else "NA"
            parts.append(
                f"[CHUNK {rank}] "
                f"doc_id={metadata.get('doc_id') or metadata.get('source_doc_id','unknown')} "
                f"chunk_id={metadata.get('chunk_id','unknown')} "
                f"chunk_index={metadata.get('chunk_index','NA')} "
                f"retrieval_score={score_txt}"
            )
        parts.append(text.strip())
        parts.append("")
    context = "\n".join(parts).strip()
    if max_chars is not None and len(context) > max_chars:
        context = context[:max_chars] + "\n...[truncated]"
    return context
