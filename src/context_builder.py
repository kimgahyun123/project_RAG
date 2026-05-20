# src/context_builder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass
class ContextChunk:
    rank: int
    doc_id: str
    chunk_id: str
    chunk_index: int | None
    retrieval_score: float | None
    text: str


@dataclass
class AssembledContext:
    chunks: list[ContextChunk]
    detector_context: str
    answer_context: str


def _unpack(item: Any):
    if isinstance(item, tuple) and len(item) >= 2:
        return item[0], item[1]
    return item, None


def _meta_value(metadata: dict, *keys: str, default: str = "unknown") -> str:
    for key in keys:
        value = metadata.get(key)
        if value not in (None, ""):
            return str(value)
    return default


def _to_int_or_none(value: Any) -> int | None:
    try:
        if value in (None, "", "NA"):
            return None
        return int(value)
    except Exception:
        return None


def _to_float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def assemble_context(
    results: Sequence[Any],
    include_metadata: bool = True,
    max_chars: int | None = None,
) -> AssembledContext:
    chunks: list[ContextChunk] = []
    detector_parts: list[str] = []
    answer_parts: list[str] = []

    for rank, item in enumerate(results, start=1):
        doc, score = _unpack(item)
        metadata = getattr(doc, "metadata", {}) or {}
        text = (getattr(doc, "page_content", "") or "").strip()

        doc_id = _meta_value(metadata, "doc_id", "source_doc_id", "source")
        chunk_id = _meta_value(metadata, "chunk_id", default=f"{doc_id}_rank_{rank}")
        chunk_index = _to_int_or_none(metadata.get("chunk_index"))
        retrieval_score = _to_float_or_none(score)

        chunk = ContextChunk(
            rank=rank,
            doc_id=doc_id,
            chunk_id=chunk_id,
            chunk_index=chunk_index,
            retrieval_score=retrieval_score,
            text=text,
        )
        chunks.append(chunk)

        if include_metadata:
            score_txt = f"{retrieval_score:.4f}" if retrieval_score is not None else "NA"
            idx_txt = chunk_index if chunk_index is not None else "NA"
            detector_parts.append(
                f"[CHUNK {rank}] doc_id={doc_id} chunk_id={chunk_id} "
                f"chunk_index={idx_txt} retrieval_score={score_txt}"
            )
        detector_parts.append(text)
        detector_parts.append("")

        answer_parts.append(text)
        answer_parts.append("")

    detector_context = "\n".join(detector_parts).strip()
    answer_context = "\n".join(answer_parts).strip()

    if max_chars is not None and len(detector_context) > max_chars:
        detector_context = detector_context[:max_chars] + "\n...[truncated]"
    if max_chars is not None and len(answer_context) > max_chars:
        answer_context = answer_context[:max_chars] + "\n...[truncated]"

    return AssembledContext(
        chunks=chunks,
        detector_context=detector_context,
        answer_context=answer_context,
    )


def build_context(
    results: Sequence[Any],
    include_metadata: bool = True,
    max_chars: int | None = None,
) -> str:
    return assemble_context(
        results=results,
        include_metadata=include_metadata,
        max_chars=max_chars,
    ).detector_context
