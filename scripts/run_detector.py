#!/usr/bin/env python3
# scripts/run_detector.py

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


import argparse
import re
import contextlib
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.vector_store import VectorStoreManager
from src.context_builder import build_context
from src.assembly_detector import AssemblyContextDetector


TOTAL_RULES = 23

def safe_list(value):
    """
    callable이면 실행하지 않는다.
    result.matched_rules가 내부 함수로 잡히는 경우가 있어서
    리스트 변환 시 callable을 호출하면 안 된다.
    """
    if value is None:
        return []

    if callable(value):
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, (tuple, set)):
        return list(value)

    if isinstance(value, str):
        return [value]

    try:
        return list(value)
    except TypeError:
        return [value]


def rule_sort_key(rule_id):
    """
    A1~A18, B1~B5 순서 정렬.
    """
    rule_id = str(rule_id)

    m = re.match(r"([A-Z]+)(\d+)", rule_id)
    if not m:
        return (99, 999, rule_id)

    prefix, num = m.group(1), int(m.group(2))
    prefix_order = {"A": 0, "B": 1}
    return (prefix_order.get(prefix, 99), num, rule_id)


def get_rule_matches(result):
    rule_matches = getattr(result, "rule_matches", {}) or {}
    if not isinstance(rule_matches, dict):
        return {}
    return rule_matches


def get_matched_rules(result):
    """
    result.matched_rules를 직접 쓰지 않고,
    result.rule_matches에서 matched=True인 룰만 안정적으로 추출한다.
    """
    rule_matches = get_rule_matches(result)

    matched = []
    for rule_id, match in rule_matches.items():
        if bool(getattr(match, "matched", False)):
            matched.append(rule_id)

    return sorted(matched, key=rule_sort_key)


def _rule_group(result, rule_id):
    match = get_rule_matches(result).get(rule_id)
    return getattr(match, "group", "") if match else ""


def _rule_type_from_group(group):
    if group == "retrieval_structure":
        return "structure"
    if group == "judge_verification":
        return "verification"
    return "semantic"


def get_active_groups(result):
    groups = []
    for rid in get_matched_rules(result):
        group = _rule_group(result, rid)
        if group and group not in groups:
            groups.append(group)
    return groups


def get_structure_rules(result):
    return [
        rid for rid in get_matched_rules(result)
        if _rule_type_from_group(_rule_group(result, rid)) == "structure"
    ]


def get_semantic_rules(result):
    return [
        rid for rid in get_matched_rules(result)
        if _rule_type_from_group(_rule_group(result, rid)) == "semantic"
    ]


def get_verification_rules(result):
    return [
        rid for rid in get_matched_rules(result)
        if _rule_type_from_group(_rule_group(result, rid)) == "verification"
    ]


def unpack_result_item(item: Any) -> Tuple[Any, float | None]:
    if isinstance(item, tuple) and len(item) >= 2:
        return item[0], float(item[1])
    return item, None


def short_text(text: str, limit: int = 180) -> str:
    text = " ".join(str(text or "").split())
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def badge_label(label: str) -> str:
    label = (label or "").lower()
    if label == "malicious":
        return "🚨 MALICIOUS"
    if label == "suspicious":
        return "⚠️  SUSPICIOUS"
    if label == "benign":
        return "✅ BENIGN"
    return f"❔ {label.upper()}"


def status_bool(value: bool) -> str:
    return "✅ True" if value else "— False"


def print_line(title: str = "") -> None:
    if title:
        print("\n" + "=" * 90)
        print(title)
        print("=" * 90)
    else:
        print("-" * 90)


def summarize_rules(result) -> Dict[str, List[str]]:
    return {
        "structure": get_structure_rules(result),
        "semantic": get_semantic_rules(result),
        "verification": get_verification_rules(result),
    }


def get_strong_rules(detector: AssemblyContextDetector, result) -> List[str]:
    if hasattr(detector, "_strong_attack_rule_ids"):
        try:
            return detector._strong_attack_rule_ids(result.rule_matches)
        except Exception:
            return []
    return []


def print_verdict(result, detector: AssemblyContextDetector) -> None:
    strong_rules = get_strong_rules(detector, result)

    print_line("📌 FINAL VERDICT")
    print(f"판정          : {badge_label(result.label)}")
    print(f"사유          : {result.reason}")
    print()
    print(f"LLM Judge     : {badge_label(result.llm_judge_label)}")
    print(f"Judge 이유    : {result.llm_judge_reason}")
    print()
    print(f"강한 공격 룰  : {strong_rules if strong_rules else '없음'}")
    print(f"Leave-one-out : {status_bool(result.ablation_verified)}")
    print(f"Source-set    : {status_bool(result.source_set_verified)}")


def print_query_info(query: str, k: int) -> None:
    print_line("🔎 QUERY")
    print(f"Top-k         : {k}")
    print(f"질의          : {query}")


def print_top_chunks(results) -> None:
    print_line("📚 RETRIEVED TOP-K CHUNKS")

    for rank, item in enumerate(results, start=1):
        doc, score = unpack_result_item(item)
        meta = getattr(doc, "metadata", {}) or {}
        text = getattr(doc, "page_content", "") or ""

        doc_id = meta.get("doc_id") or meta.get("source_doc_id") or "unknown"
        chunk_id = meta.get("chunk_id") or "unknown"
        chunk_index = meta.get("chunk_index", "NA")
        score_text = f"{score:.4f}" if score is not None else "NA"

        print(f"[Rank {rank}] score={score_text}")
        print(f"  doc_id      : {doc_id}")
        print(f"  chunk_id    : {chunk_id}")
        print(f"  chunk_index : {chunk_index}")
        print(f"  snippet     : {short_text(text)}")
        print()


def print_rule_summary(result) -> None:
    groups = summarize_rules(result)

    print_line("🧩 RULE SUMMARY")
    matched_rules = get_matched_rules(result)
    print(f"Matched rules : {len(matched_rules)}/{TOTAL_RULES}")
    active_groups = get_active_groups(result)
    print(f"Active groups : {', '.join(active_groups) if active_groups else '없음'}")
    print()

    print(f"구조 룰       : {groups['structure'] if groups['structure'] else '없음'}")
    print(f"의미 룰       : {groups['semantic'] if groups['semantic'] else '없음'}")
    print(f"검증 룰       : {groups['verification'] if groups['verification'] else '없음'}")


def print_rule_details(result) -> None:
    print_line("📋 MATCHED RULE DETAILS")

    matched_rules = get_matched_rules(result)
    if not matched_rules:
        print("탐지된 룰 없음")
        return

    for rule_id in matched_rules:
        match = result.rule_matches.get(rule_id)
        if not match:
            continue

        print(f"{rule_id:>3} | {match.group:<25} | {match.name}")
        for e in match.evidence[:3]:
            print(f"      - {e}")
        if len(match.evidence) > 3:
            print(f"      - ... +{len(match.evidence) - 3} more")
        print()


def print_ablation(result) -> None:
    print_line("🧪 ABLATION VERIFICATION")

    print(f"핵심 의심 문서        : {result.verified_suspect_doc_ids if result.verified_suspect_doc_ids else '없음'}")
    print(f"의심 문서 조합        : {result.verified_suspect_doc_sets if result.verified_suspect_doc_sets else '없음'}")
    print(f"전체 suspect docs     : {result.suspect_source_doc_ids if result.suspect_source_doc_ids else '없음'}")
    print(f"evidence chunks       : {result.evidence_chunk_ids if result.evidence_chunk_ids else '없음'}")

    if result.leave_one_out_results:
        print("\n[Leave-one-out]")
        for row in result.leave_one_out_results:
            mark = "✅" if row.get("verified") else "—"
            print(
                f"{mark} remove {row.get('removed_doc_id')}: "
                f"risk_drop={row.get('risk_drop')}, "
                f"semantic_drop={row.get('semantic_drop')}"
            )
            print(f"   dropped_rules: {row.get('dropped_rules')}")
            print(f"   reason       : {row.get('verification_reason')}")

    if result.source_set_ablation_results:
        print("\n[Source-set ablation]")
        for row in result.source_set_ablation_results:
            mark = "✅" if row.get("verified") else "—"
            print(
                f"{mark} remove {row.get('removed_doc_ids')}: "
                f"risk_drop={row.get('risk_drop')}, "
                f"semantic_drop={row.get('semantic_drop')}"
            )
            print(f"   dropped_rules: {row.get('dropped_rules')}")
            print(f"   reason       : {row.get('verification_reason')}")


def print_interpretation(result, detector: AssemblyContextDetector) -> None:
    strong_rules = get_strong_rules(detector, result)

    print_line("🧠 INTERPRETATION GUIDE")

    if result.label == "malicious":
        print("해석: 악성 조립 문맥으로 볼 수 있음.")
        print("이유: LLM Judge warning + 강한 공격 룰 + ablation 검증이 함께 충족됨.")
        if result.verified_suspect_doc_ids:
            print(f"권장 격리 후보: {result.verified_suspect_doc_ids}")
        elif result.verified_suspect_doc_sets:
            print(f"권장 검토 조합: {result.verified_suspect_doc_sets}")

    elif result.label == "suspicious":
        print("해석: 위험 후보지만 악성 확정은 아님.")
        print("확인할 것:")
        print("  1) LLM Judge가 suspicious/malicious인지")
        print("  2) B3, A4, A5, A7, A14, B5, A18 같은 강한 룰이 있는지")
        print("  3) ablation으로 특정 문서 제거 시 위험 신호가 크게 줄었는지")
        print(f"현재 강한 공격 룰: {strong_rules if strong_rules else '없음 또는 부족'}")

    else:
        print("해석: 정상 문맥으로 볼 수 있음.")
        print("약한 룰이 일부 잡혀도 LLM Judge가 benign이고 강한 공격 룰이 없으면 정상으로 판단.")


def make_json_summary(query: str, result, detector: AssemblyContextDetector) -> Dict[str, Any]:
    return {
        "query": query,
        "label": result.label,
        "reason": result.reason,
        "llm_judge_label": result.llm_judge_label,
        "llm_judge_reason": result.llm_judge_reason,
        "matched_rules": get_matched_rules(result),
        "strong_attack_rules": get_strong_rules(detector, result),
        "structure_rules": get_structure_rules(result),
        "semantic_rules": get_semantic_rules(result),
        "verification_rules": get_verification_rules(result),
        "active_groups": get_active_groups(result),
        "suspect_source_doc_ids": result.suspect_source_doc_ids,
        "evidence_chunk_ids": result.evidence_chunk_ids,
        "ablation_verified": result.ablation_verified,
        "verified_suspect_doc_ids": result.verified_suspect_doc_ids,
        "source_set_verified": result.source_set_verified,
        "verified_suspect_doc_sets": result.verified_suspect_doc_sets,
        "leave_one_out_results": result.leave_one_out_results,
        "source_set_ablation_results": result.source_set_ablation_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RAG Assembly Context Detector with visual summary."
    )
    parser.add_argument("query", nargs="+", help="질의문. 따옴표로 감싸서 입력 권장.")
    parser.add_argument("-k", "--top-k", type=int, default=5)
    parser.add_argument("--index-path", default="data/faiss_index")
    parser.add_argument("--no-judge", action="store_true", help="LLM Judge 비활성화")
    parser.add_argument("--raw", action="store_true", help="기존 detector 원본 출력도 표시")
    parser.add_argument("--show-chunks", action="store_true", help="검색된 Top-k 청크 요약 표시")
    parser.add_argument("--save-json", default=None, help="결과 요약 JSON 저장 경로")

    args = parser.parse_args()
    query = " ".join(args.query).strip()

    vm = VectorStoreManager(index_path=args.index_path)
    vm.load()

    results = vm.search(query, k=args.top_k)
    context = build_context(results, include_metadata=True)

    detector = AssemblyContextDetector(
        enable_llm_judge=not args.no_judge,
        enable_leave_one_out=True,
        enable_source_set_ablation=True,
    )

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = detector.detect(
            query=query,
            results=results,
            assembled_context=context,
        )

    raw_output = buffer.getvalue()

    print_query_info(query, args.top_k)
    print_verdict(result, detector)
    print_rule_summary(result)
    print_ablation(result)
    print_interpretation(result, detector)

    if args.show_chunks:
        print_top_chunks(results)

    if args.raw:
        print_line("RAW DETECTOR OUTPUT")
        print(raw_output)

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary = make_json_summary(query, result, detector)
        out_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print_line()
        print(f"JSON 저장 완료: {out_path}")


if __name__ == "__main__":
    main()
