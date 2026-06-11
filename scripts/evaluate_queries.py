# scripts/evaluate_queries.py
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag import run_query


def _labels_by_round(result: dict[str, Any]) -> list[str]:
    return [row["detection"].label for row in result.get("history", [])]


def _scores_by_round(result: dict[str, Any]) -> list[float]:
    return [row["detection"].risk_score for row in result.get("history", [])]


def _via_by_round(result: dict[str, Any]) -> list[str]:
    return [row["detection"].via for row in result.get("history", [])]


def _rule_labels_by_round(result: dict[str, Any]) -> list[str]:
    return [row["detection"].rule_label for row in result.get("history", [])]


def _rule_scores_by_round(result: dict[str, Any]) -> list[float]:
    return [row["detection"].rule_score for row in result.get("history", [])]


def _judge_labels_by_round(result: dict[str, Any]) -> list[str]:
    return [row["detection"].judge_label for row in result.get("history", [])]


def _judge_scores_by_round(result: dict[str, Any]) -> list[float]:
    return [row["detection"].judge_score for row in result.get("history", [])]


def _reasons_by_round(result: dict[str, Any]) -> list[str]:
    return [row["detection"].reason for row in result.get("history", [])]


def _judge_reasons_by_round(result: dict[str, Any]) -> list[str]:
    return [row["detection"].judge_reason for row in result.get("history", [])]


def _quarantine_by_round(result: dict[str, Any]) -> list[list[str]]:
    return [row["detection"].quarantine_doc_ids for row in result.get("history", [])]


def _top_docs_by_round(result: dict[str, Any]) -> list[list[str]]:
    out = []
    for row in result.get("history", []):
        docs = []
        for item in row.get("results", []):
            doc = item[0] if isinstance(item, tuple) and len(item) >= 2 else item
            meta = getattr(doc, "metadata", {}) or {}
            docs.append(str(meta.get("doc_id") or meta.get("source_doc_id") or "unknown"))
        out.append(docs)
    return out


def _top_chunks_by_round(result: dict[str, Any]) -> list[list[dict[str, Any]]]:
    out = []
    for row in result.get("history", []):
        chunks = []
        for rank, item in enumerate(row.get("results", []), start=1):
            doc = item[0] if isinstance(item, tuple) and len(item) >= 2 else item
            score = item[1] if isinstance(item, tuple) and len(item) >= 2 else None
            meta = getattr(doc, "metadata", {}) or {}
            chunks.append({
                "rank": rank,
                "doc_id": str(meta.get("doc_id") or meta.get("source_doc_id") or "unknown"),
                "chunk_id": str(meta.get("chunk_id") or "unknown"),
                "chunk_index": meta.get("chunk_index", "NA"),
                "score": float(score) if score is not None else None,
            })
        out.append(chunks)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/eval_queries.csv")
    parser.add_argument("--output", default="results/eval_summary.csv")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--index-path", default="data/faiss_index")
    parser.add_argument("--no-judge", action="store_true")
    args = parser.parse_args()

    rows = []
    with Path(args.input).open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for i, row in enumerate(rows, start=1):
        query_id = row.get("query_id", "")
        kind = row.get("kind", "")
        query = row.get("query", "")

        print("\n" + "=" * 80)
        print(f"[{i}/{len(rows)}] query_id={query_id} kind={kind}")
        print(query)

        try:
            result = run_query(
                query=query,
                k=args.k,
                index_path=args.index_path,
                use_judge=not args.no_judge,
                max_rounds=args.max_rounds,
                use_quarantine_db=False,
                generate_answer=False,
            )

            labels = _labels_by_round(result)

            summary_rows.append({
                "query_id": query_id,
                "kind": kind,
                "status": result.get("status", ""),
                "rounds": len(result.get("history", [])),
                "final_label": labels[-1] if labels else "",
                "quarantined_doc_ids": json.dumps(result.get("quarantined_doc_ids", []), ensure_ascii=False),
                "labels_by_round": json.dumps(labels, ensure_ascii=False),
                "scores_by_round": json.dumps(_scores_by_round(result), ensure_ascii=False),
                "via_by_round": json.dumps(_via_by_round(result), ensure_ascii=False),
                "rule_labels_by_round": json.dumps(_rule_labels_by_round(result), ensure_ascii=False),
                "rule_scores_by_round": json.dumps(_rule_scores_by_round(result), ensure_ascii=False),
                "judge_labels_by_round": json.dumps(_judge_labels_by_round(result), ensure_ascii=False),
                "judge_scores_by_round": json.dumps(_judge_scores_by_round(result), ensure_ascii=False),
                "reasons_by_round": json.dumps(_reasons_by_round(result), ensure_ascii=False),
                "judge_reasons_by_round": json.dumps(_judge_reasons_by_round(result), ensure_ascii=False),
                "quarantine_by_round": json.dumps(_quarantine_by_round(result), ensure_ascii=False),
                "top_docs_by_round": json.dumps(_top_docs_by_round(result), ensure_ascii=False),
                "top_chunks_by_round": json.dumps(_top_chunks_by_round(result), ensure_ascii=False),
                "block_reason": result.get("block_reason", ""),
                "query": query,
            })

        except Exception as exc:
            summary_rows.append({
                "query_id": query_id,
                "kind": kind,
                "status": "error",
                "rounds": 0,
                "final_label": "error",
                "quarantined_doc_ids": "[]",
                "labels_by_round": "[]",
                "scores_by_round": "[]",
                "via_by_round": "[]",
                "rule_labels_by_round": "[]",
                "rule_scores_by_round": "[]",
                "judge_labels_by_round": "[]",
                "judge_scores_by_round": "[]",
                "reasons_by_round": "[]",
                "judge_reasons_by_round": "[]",
                "quarantine_by_round": "[]",
                "top_docs_by_round": "[]",
                "top_chunks_by_round": "[]",
                "block_reason": repr(exc),
                "query": query,
            })

    fieldnames = [
        "query_id",
        "kind",
        "status",
        "rounds",
        "final_label",
        "quarantined_doc_ids",
        "labels_by_round",
        "scores_by_round",
        "via_by_round",
        "rule_labels_by_round",
        "rule_scores_by_round",
        "judge_labels_by_round",
        "judge_scores_by_round",
        "reasons_by_round",
        "judge_reasons_by_round",
        "quarantine_by_round",
        "top_docs_by_round",
        "top_chunks_by_round",
        "block_reason",
        "query",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n" + "=" * 80)
    print(f"saved: {out_path}")
    print(f"total: {len(summary_rows)}")
    print(f"by kind: {Counter(r['kind'] for r in summary_rows)}")
    print(f"status: {Counter(r['status'] for r in summary_rows)}")
    print(f"final_label: {Counter(r['final_label'] for r in summary_rows)}")
    print(f"normal problems: {[r['query_id'] for r in summary_rows if r['kind'] == 'normal' and (r['status'] != 'answered' or r['final_label'] != 'clean')]}")
    print(f"attack problems: {[r['query_id'] for r in summary_rows if r['kind'] == 'attack' and r['status'] != 'answered']}")
    print(f"normal quarantined: {[r['query_id'] for r in summary_rows if r['kind'] == 'normal' and r['quarantined_doc_ids'] != '[]']}")


if __name__ == "__main__":
    main()
