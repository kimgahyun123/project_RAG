# src/rag.py
from __future__ import annotations

import argparse
from typing import Any, Iterable, List, Sequence, Set, Tuple

from src.assembly_detector import AssemblyDetector
from src.context_builder import assemble_context
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


def _filter_quarantined(results: Sequence[Any], quarantined_doc_ids: Set[str]) -> List[Any]:
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
    quarantined_doc_ids: Set[str],
    overfetch_factor: int = 4,
) -> List[Any]:
    raw_k = max(k * overfetch_factor, k + len(quarantined_doc_ids) * 2)
    raw_results = vm.search(query, k=raw_k)
    active_results = _filter_quarantined(raw_results, quarantined_doc_ids)
    return active_results[:k]


def run_query(
    query: str,
    k: int = 5,
    index_path: str = "data/faiss_index",
    use_judge: bool = False,
    max_rounds: int = 3,
):
    vm = VectorStoreManager(index_path=index_path)
    vm.load()

    detector = AssemblyDetector(
        use_llm_judge=use_judge,
    )

    quarantined_doc_ids: Set[str] = set()
    history = []

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
            "quarantined_doc_ids": sorted(quarantined_doc_ids),
        })

        print("\n" + "=" * 60)
        print(f"[Round {round_no}] detection={detection.label} score={detection.risk_score:.3f}")
        print(f"reason: {detection.reason}")
        print(f"quarantine candidates: {detection.quarantine_doc_ids}")

        if detection.label == "clean":
            print("\n[Final context]")
            print(assembled.answer_context)
            return {
                "status": "answered",
                "results": results,
                "assembled": assembled,
                "detection": detection,
                "history": history,
                "quarantined_doc_ids": sorted(quarantined_doc_ids),
            }

        new_doc_ids = [
            doc_id for doc_id in detection.quarantine_doc_ids
            if doc_id and doc_id != "unknown" and doc_id not in quarantined_doc_ids
        ]

        if not new_doc_ids:
            return {
                "status": "blocked",
                "results": results,
                "assembled": assembled,
                "detection": detection,
                "history": history,
                "quarantined_doc_ids": sorted(quarantined_doc_ids),
                "block_reason": "Detector found risk but no new document could be quarantined.",
            }

        quarantined_doc_ids.update(new_doc_ids)

    return {
        "status": "blocked",
        "results": history[-1]["results"] if history else [],
        "assembled": history[-1]["assembled"] if history else None,
        "detection": history[-1]["detection"] if history else None,
        "history": history,
        "quarantined_doc_ids": sorted(quarantined_doc_ids),
        "block_reason": "Max detection rounds reached.",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("query", nargs="*", help="질의문")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--index-path", default="data/faiss_index")
    p.add_argument("--judge", action="store_true")
    p.add_argument("--max-rounds", type=int, default=3)
    args = p.parse_args()

    query = " ".join(args.query).strip()
    if not query:
        query = input("Query: ").strip()

    result = run_query(
        query,
        k=args.k,
        index_path=args.index_path,
        use_judge=args.judge,
        max_rounds=args.max_rounds,
    )

    print("\n" + "=" * 60)
    print(f"status: {result['status']}")
    print(f"quarantined_doc_ids: {result['quarantined_doc_ids']}")
    if result.get("block_reason"):
        print(f"block_reason: {result['block_reason']}")


if __name__ == "__main__":
    main()
