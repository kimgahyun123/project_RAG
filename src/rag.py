# src/rag.py
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv

from src.assembly_detector import AssemblyDetector, AssemblyDetectionResult
from src.context_builder import AssembledContext, assemble_context
from src.quarantine import QuarantineStore
from src.query_log import append_query_run
from src.vector_store import VectorStoreManager


def _doc_id_of_result(item: Any) -> str:
    doc = item[0] if isinstance(item, tuple) and len(item) >= 2 else item
    metadata = getattr(doc, "metadata", {}) or {}
    return str(
        metadata.get("doc_id")
        or metadata.get("source_doc_id")
        or metadata.get("source")
        or "unknown"
    )


def _chunk_id_of_result(item: Any) -> str:
    doc = item[0] if isinstance(item, tuple) and len(item) >= 2 else item
    metadata = getattr(doc, "metadata", {}) or {}
    return str(metadata.get("chunk_id") or "unknown")


def _score_of_result(item: Any) -> float | None:
    if isinstance(item, tuple) and len(item) >= 2:
        try:
            return float(item[1])
        except Exception:
            return None
    return None


def _filter_quarantined(results: Sequence[Any], quarantined_doc_ids: set[str]) -> list[Any]:
    if not quarantined_doc_ids:
        return list(results)
    return [
        item for item in results
        if _doc_id_of_result(item) not in quarantined_doc_ids
    ]


def _search_active(
    vm: VectorStoreManager,
    query: str,
    k: int,
    quarantined_doc_ids: set[str],
    overfetch_factor: int = 4,
) -> list[Any]:
    raw_k = max(k * overfetch_factor, k + len(quarantined_doc_ids) * 5, 20)
    raw_results = vm.search(query, k=raw_k)
    active_results = _filter_quarantined(raw_results, quarantined_doc_ids)
    return active_results[:k]


def _make_answer_llm():
    load_dotenv()
    model = (
        os.getenv("ANTHROPIC_ANSWER_MODEL")
        or os.getenv("ANTHROPIC_MODEL")
        or "claude-opus-4-5-20251101"
    )

    try:
        from langchain_anthropic import ChatAnthropic
    except Exception as exc:
        raise ImportError(
            "langchain-anthropic is required for answer generation. "
            "Install with: pip install langchain-anthropic"
        ) from exc

    return ChatAnthropic(
        model=model,
        temperature=0,
        max_tokens=1200,
    )


def _generate_answer(query: str, context: str, llm) -> str:
    prompt = f"""
You are a legal RAG assistant.

Answer the user's question using only the provided context.
If the context is insufficient, say that the context is insufficient.
Do not use quarantined or removed documents.
Be concise and legally careful.

Question:
{query}

Clean retrieved context:
{context}

Answer:
""".strip()

    response = llm.invoke(prompt)
    return str(getattr(response, "content", response)).strip()


def _print_round(round_no: int, detection: AssemblyDetectionResult, results: Sequence[Any]) -> None:
    rule_info = f"{detection.rule_label}:{detection.rule_score:.3f}"
    print("\n" + "=" * 60)
    print(
        f"[Round {round_no}] detection={detection.label} "
        f"score={detection.risk_score:.3f} via={detection.via} rule={rule_info}"
    )
    print(f"reason: {detection.reason}")
    print(f"quarantine candidates: {detection.quarantine_doc_ids}")
    print("top-k:")
    for rank, item in enumerate(results, start=1):
        score = _score_of_result(item)
        score_txt = f"{score:.4f}" if score is not None else "NA"
        print(
            f"  {rank}. doc_id={_doc_id_of_result(item)} "
            f"chunk_id={_chunk_id_of_result(item)} score={score_txt}"
        )


def _append_history_csv(history: list[dict[str, Any]], status: str, query: str) -> None:
    out_path = Path("results/rag_history.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    exists = out_path.exists()
    with out_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "query",
                "status",
                "round",
                "label",
                "score",
                "via",
                "rule_label",
                "rule_score",
                "quarantine_doc_ids",
                "top_doc_ids",
                "reason",
            ],
        )
        if not exists:
            writer.writeheader()

        for row in history:
            detection = row["detection"]
            results = row["results"]
            writer.writerow({
                "query": query,
                "status": status,
                "round": row["round"],
                "label": detection.label,
                "score": detection.risk_score,
                "via": detection.via,
                "rule_label": detection.rule_label,
                "rule_score": detection.rule_score,
                "quarantine_doc_ids": json.dumps(detection.quarantine_doc_ids, ensure_ascii=False),
                "top_doc_ids": json.dumps([_doc_id_of_result(x) for x in results], ensure_ascii=False),
                "reason": detection.reason,
            })


def run_query(
    query: str,
    k: int = 5,
    index_path: str = "data/faiss_index",
    use_judge: bool = True,
    max_rounds: int = 10,
    use_quarantine_db: bool = False,
    quarantine_db_path: str = "data/quarantine.sqlite3",
    generate_answer: bool = True,
) -> dict[str, Any]:
    vm = VectorStoreManager(index_path=index_path)
    vm.load()

    detector = AssemblyDetector(use_llm_judge=use_judge)
    answer_llm = _make_answer_llm() if generate_answer else None

    qstore = QuarantineStore(quarantine_db_path) if use_quarantine_db else None
    quarantined_doc_ids: set[str] = set(qstore.active_doc_ids() if qstore else [])
    run_quarantined_doc_ids: set[str] = set()

    history: list[dict[str, Any]] = []

    try:
        for round_no in range(1, max_rounds + 1):
            results = _search_active(
                vm=vm,
                query=query,
                k=k,
                quarantined_doc_ids=quarantined_doc_ids,
            )

            assembled = assemble_context(results)
            detection = detector.detect(query=query, assembled=assembled)

            history.append({
                "round": round_no,
                "results": results,
                "assembled": assembled,
                "detection": detection,
                "quarantined_doc_ids_before": sorted(quarantined_doc_ids),
            })

            _print_round(round_no, detection, results)

            if detection.label == "clean":
                print("\n[Final context]")
                print(assembled.answer_context)

                final_answer = None
                if generate_answer and answer_llm is not None:
                    final_answer = _generate_answer(
                        query=query,
                        context=assembled.answer_context,
                        llm=answer_llm,
                    )
                    print("\n[Final answer]")
                    print(final_answer)

                _append_history_csv(history, "answered", query)
                return {
                    "status": "answered",
                    "results": results,
                    "assembled": assembled,
                    "detection": detection,
                    "history": history,
                    "quarantined_doc_ids": sorted(quarantined_doc_ids),
                    "run_quarantined_doc_ids": sorted(run_quarantined_doc_ids),
                    "answer": final_answer,
                }

            new_doc_ids = [
                doc_id for doc_id in detection.quarantine_doc_ids
                if doc_id and doc_id != "unknown" and doc_id not in quarantined_doc_ids
            ]

            if not new_doc_ids:
                _append_history_csv(history, "blocked", query)
                return {
                    "status": "blocked",
                    "results": results,
                    "assembled": assembled,
                    "detection": detection,
                    "history": history,
                    "quarantined_doc_ids": sorted(quarantined_doc_ids),
                    "run_quarantined_doc_ids": sorted(run_quarantined_doc_ids),
                    "block_reason": (
                        "Context was suspicious/malicious, "
                        "but no new document could be quarantined."
                    ),
                }

            if qstore:
                for doc_id in new_doc_ids:
                    qstore.quarantine_doc(
                        doc_id=doc_id,
                        label=detection.label,
                        reason=detection.reason,
                        query=query,
                    )

            quarantined_doc_ids.update(new_doc_ids)
            run_quarantined_doc_ids.update(new_doc_ids)

        _append_history_csv(history, "blocked", query)
        return {
            "status": "blocked",
            "results": history[-1]["results"] if history else [],
            "assembled": history[-1]["assembled"] if history else None,
            "detection": history[-1]["detection"] if history else None,
            "history": history,
            "quarantined_doc_ids": sorted(quarantined_doc_ids),
            "run_quarantined_doc_ids": sorted(run_quarantined_doc_ids),
            "block_reason": "Max detection rounds reached.",
        }
    finally:
        if qstore:
            qstore.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("query", nargs="*", help="질의문")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--index-path", default="data/faiss_index")
    p.add_argument("--max-rounds", type=int, default=10)
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--no-answer", action="store_true")
    p.add_argument("--use-quarantine-db", action="store_true")
    p.add_argument("--quarantine-db-path", default="data/quarantine.sqlite3")
    p.add_argument("--reset-quarantine", action="store_true")
    args = p.parse_args()

    if args.reset_quarantine:
        qstore = QuarantineStore(args.quarantine_db_path)
        qstore.reset()
        qstore.close()
        print(f"Reset quarantine DB: {args.quarantine_db_path}")
        if not args.query:
            return

    query = " ".join(args.query).strip()
    if not query:
        query = input("Query: ").strip()

    result = run_query(
        query=query,
        k=args.k,
        index_path=args.index_path,
        use_judge=not args.no_judge,
        max_rounds=args.max_rounds,
        use_quarantine_db=args.use_quarantine_db,
        quarantine_db_path=args.quarantine_db_path,
        generate_answer=not args.no_answer,
    )

    append_query_run(query=query, result=result)

    print("\n" + "=" * 60)
    print(f"status: {result['status']}")
    print(f"quarantined_doc_ids: {result.get('run_quarantined_doc_ids', result.get('quarantined_doc_ids', []))}")
    if result.get("block_reason"):
        print(f"block_reason: {result['block_reason']}")


if __name__ == "__main__":
    main()
