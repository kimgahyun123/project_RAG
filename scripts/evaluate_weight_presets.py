from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


ATTACK_EXPECTED_DOCS = {
    "1": {"kb_0001", "kb_0002", "kb_0003", "kb_0004", "kb_0005"},
    "22": {"kb_0006", "kb_0007", "kb_0008", "kb_0009", "kb_0010"},
    "33": {"kb_0011", "kb_0012", "kb_0013", "kb_0014", "kb_0015"},
}

ATTACK_DOC_IDS = set().union(*ATTACK_EXPECTED_DOCS.values())
ATTACK_QUERY_IDS = set(ATTACK_EXPECTED_DOCS.keys())


RULE_GROUPS = {
    # 구조 기반 룰
    "A1": "structure",
    "A2": "structure",
    "A3": "structure",
    "A4": "structure",
    "A5": "structure",

    # 표현/위치 기반 룰
    "B1": "lexical",
    "B2": "lexical",
    "B3": "lexical",
    "B4": "lexical",
    "B5": "lexical",
    "B6": "lexical",
    "B7": "lexical",

    # 의미 기반 룰
    "C1": "semantic",
    "C2": "semantic",
    "C3": "semantic",
    "C4": "semantic",
    "C5": "semantic",
    "C6": "semantic",
    "C7": "semantic",
    "C8": "semantic",
}


WEIGHT_PRESETS = {
    "baseline": {
        "structure": 1.0,
        "lexical": 1.0,
        "semantic": 1.0,
    },
    "structure_high": {
        "structure": 1.5,
        "lexical": 1.0,
        "semantic": 1.0,
    },
    "structure_low": {
        "structure": 0.7,
        "lexical": 1.0,
        "semantic": 1.0,
    },
    "lexical_high": {
        "structure": 1.0,
        "lexical": 1.5,
        "semantic": 1.0,
    },
    "lexical_low": {
        "structure": 1.0,
        "lexical": 0.7,
        "semantic": 1.0,
    },
    "semantic_high": {
        "structure": 1.0,
        "lexical": 1.0,
        "semantic": 1.5,
    },
    "semantic_low": {
        "structure": 1.0,
        "lexical": 1.0,
        "semantic": 0.7,
    },
    "sensitive": {
        "structure": 1.3,
        "lexical": 1.3,
        "semantic": 1.3,
    },
    "conservative": {
        "structure": 0.7,
        "lexical": 0.7,
        "semantic": 0.9,
    },
}


def read_queries(path: Path, limit: int | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query_id = str(row.get("query_id") or row.get("id") or "").strip()
            query = str(row.get("query") or row.get("question") or "").strip()
            kind = str(row.get("kind") or "").strip()

            if not query_id or not query:
                continue

            if not kind:
                kind = "attack" if query_id in ATTACK_QUERY_IDS else "normal"

            rows.append({
                "query_id": query_id,
                "kind": kind,
                "query": query,
            })

            if limit and len(rows) >= limit:
                break

    return rows


def safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def parse_json_list(value: str) -> list[Any]:
    try:
        data = json.loads(value)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def normalize_doc_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                value = parsed
            else:
                value = [value]
        except Exception:
            value = [value]
    return sorted({str(x) for x in value if x})


def build_weighted_detector_class(preset_name: str):
    from src.assembly_detector import AssemblyDetector

    preset = WEIGHT_PRESETS[preset_name]

    class WeightedAssemblyDetector(AssemblyDetector):
        def _weight_multiplier(self, rule_id: str) -> float:
            group = RULE_GROUPS.get(rule_id)
            if not group:
                return 1.0
            return float(preset.get(group, 1.0))

        def _scale_hit(self, hit):
            multiplier = self._weight_multiplier(hit.rule_id)
            return replace(hit, weight=round(float(hit.weight) * multiplier, 6))

        def _scaled_hits(self, hits):
            return [self._scale_hit(hit) for hit in hits]

        def _score(self, hits):
            return super()._score(self._scaled_hits(hits))

        def _choose_quarantine_docs(self, hits, chunks):
            return super()._choose_quarantine_docs(self._scaled_hits(hits), chunks)

    return WeightedAssemblyDetector


def patch_rag_for_experiment(preset_name: str):
    import src.rag as rag

    rag.AssemblyDetector = build_weighted_detector_class(preset_name)

    # 실험 결과가 관리자 대시보드용 rag_history.csv에 섞이지 않게 막는다.
    rag._append_history_csv = lambda history, status, query: None

    return rag


def labels_by_round(result: dict[str, Any]) -> list[str]:
    labels = []
    for row in result.get("history", []) or []:
        detection = row.get("detection")
        labels.append(getattr(detection, "label", "unknown"))
    return labels


def rule_labels_by_round(result: dict[str, Any]) -> list[str]:
    labels = []
    for row in result.get("history", []) or []:
        detection = row.get("detection")
        labels.append(getattr(detection, "rule_label", "unknown"))
    return labels


def rule_scores_by_round(result: dict[str, Any]) -> list[float]:
    scores = []
    for row in result.get("history", []) or []:
        detection = row.get("detection")
        scores.append(float(getattr(detection, "rule_score", 0.0)))
    return scores


def quarantine_by_round(result: dict[str, Any]) -> list[list[str]]:
    out = []
    for row in result.get("history", []) or []:
        detection = row.get("detection")
        out.append(normalize_doc_ids(getattr(detection, "quarantine_doc_ids", [])))
    return out


def matched_rules_by_round(result: dict[str, Any]) -> list[list[str]]:
    out = []
    for row in result.get("history", []) or []:
        detection = row.get("detection")
        hits = getattr(detection, "matched_rules", []) or []
        out.append([f"{h.rule_id}:{h.name}:{h.weight}" for h in hits])
    return out


def run_one_query(
    preset_name: str,
    query: str,
    k: int,
    max_rounds: int,
    index_path: str,
    use_judge: bool,
) -> dict[str, Any]:
    rag = patch_rag_for_experiment(preset_name)

    return rag.run_query(
        query=query,
        k=k,
        index_path=index_path,
        use_judge=use_judge,
        max_rounds=max_rounds,
        use_quarantine_db=False,
        generate_answer=False,
    )


def evaluate_preset(
    preset_name: str,
    rows: list[dict[str, str]],
    k: int,
    max_rounds: int,
    index_path: str,
    use_judge: bool,
) -> list[dict[str, Any]]:
    detail_rows: list[dict[str, Any]] = []

    for i, row in enumerate(rows, start=1):
        query_id = row["query_id"]
        kind = row["kind"]
        query = row["query"]

        print("=" * 80)
        print(f"preset={preset_name} [{i}/{len(rows)}] query_id={query_id} kind={kind}")

        try:
            result = run_one_query(
                preset_name=preset_name,
                query=query,
                k=k,
                max_rounds=max_rounds,
                index_path=index_path,
                use_judge=use_judge,
            )

            labels = labels_by_round(result)
            rule_labels = rule_labels_by_round(result)
            rule_scores = rule_scores_by_round(result)
            q_by_round = quarantine_by_round(result)
            matched_rules = matched_rules_by_round(result)
            quarantined_docs = normalize_doc_ids(result.get("quarantined_doc_ids", []))

            final_label = labels[-1] if labels else ""
            first_label = labels[0] if labels else ""
            final_rule_label = rule_labels[-1] if rule_labels else ""
            first_rule_label = rule_labels[0] if rule_labels else ""

            expected_docs = ATTACK_EXPECTED_DOCS.get(query_id, set())
            quarantined_set = set(quarantined_docs)

            expected_hit = sorted(quarantined_set & expected_docs)
            expected_missed = sorted(expected_docs - quarantined_set)

            normal_fp_docs = []
            attack_docs_in_normal = []

            if kind == "normal":
                for doc_id in quarantined_docs:
                    if doc_id in ATTACK_DOC_IDS:
                        attack_docs_in_normal.append(doc_id)
                    else:
                        normal_fp_docs.append(doc_id)

            detail_rows.append({
                "preset": preset_name,
                "mode": "rule_judge" if use_judge else "rule_only",
                "query_id": query_id,
                "kind": kind,
                "status": result.get("status", ""),
                "rounds": len(result.get("history", []) or []),
                "first_label": first_label,
                "final_label": final_label,
                "first_rule_label": first_rule_label,
                "final_rule_label": final_rule_label,
                "rule_scores_by_round": safe_json(rule_scores),
                "labels_by_round": safe_json(labels),
                "rule_labels_by_round": safe_json(rule_labels),
                "quarantine_by_round": safe_json(q_by_round),
                "quarantined_doc_ids": safe_json(quarantined_docs),
                "expected_attack_hit": safe_json(expected_hit),
                "expected_attack_missed": safe_json(expected_missed),
                "normal_false_positive_doc_ids": safe_json(sorted(normal_fp_docs)),
                "attack_docs_quarantined_in_normal_query": safe_json(sorted(attack_docs_in_normal)),
                "matched_rules_by_round": safe_json(matched_rules),
                "block_reason": result.get("block_reason", ""),
                "query": query,
            })

        except Exception as exc:
            detail_rows.append({
                "preset": preset_name,
                "mode": "rule_judge" if use_judge else "rule_only",
                "query_id": query_id,
                "kind": kind,
                "status": "error",
                "rounds": 0,
                "first_label": "error",
                "final_label": "error",
                "first_rule_label": "error",
                "final_rule_label": "error",
                "rule_scores_by_round": "[]",
                "labels_by_round": "[]",
                "rule_labels_by_round": "[]",
                "quarantine_by_round": "[]",
                "quarantined_doc_ids": "[]",
                "expected_attack_hit": "[]",
                "expected_attack_missed": "[]",
                "normal_false_positive_doc_ids": "[]",
                "attack_docs_quarantined_in_normal_query": "[]",
                "matched_rules_by_round": "[]",
                "block_reason": repr(exc),
                "query": query,
            })

    return detail_rows


def summarize_preset(preset_name: str, detail_rows: list[dict[str, Any]], use_judge: bool) -> dict[str, Any]:
    total = len(detail_rows)
    attack_rows = [r for r in detail_rows if r["kind"] == "attack"]
    normal_rows = [r for r in detail_rows if r["kind"] == "normal"]

    by_status = Counter(r["status"] for r in detail_rows)
    by_first = Counter(r["first_label"] for r in detail_rows)
    by_final = Counter(r["final_label"] for r in detail_rows)
    by_first_rule = Counter(r["first_rule_label"] for r in detail_rows)
    by_final_rule = Counter(r["final_rule_label"] for r in detail_rows)

    attack_query_success = 0
    attack_query_misses = []
    attack_doc_misses = []

    for r in attack_rows:
        query_id = r["query_id"]
        expected = ATTACK_EXPECTED_DOCS.get(query_id, set())
        quarantined = set(parse_json_list(r["quarantined_doc_ids"]))

        if quarantined & expected:
            attack_query_success += 1
        else:
            attack_query_misses.append(query_id)

        missed = sorted(expected - quarantined)
        if missed:
            attack_doc_misses.append({
                "query_id": query_id,
                "missed": missed,
                "quarantined": sorted(quarantined),
            })

    normal_fp_queries = []
    normal_fp_docs = []
    attack_docs_in_normal = []

    for r in normal_rows:
        fp_docs = parse_json_list(r["normal_false_positive_doc_ids"])
        atk_docs = parse_json_list(r["attack_docs_quarantined_in_normal_query"])

        if fp_docs:
            normal_fp_queries.append(r["query_id"])
            normal_fp_docs.extend(fp_docs)

        if atk_docs:
            attack_docs_in_normal.append({
                "query_id": r["query_id"],
                "doc_ids": atk_docs,
            })

    avg_rounds = 0.0
    if total:
        avg_rounds = sum(int(r.get("rounds") or 0) for r in detail_rows) / total

    final_clean_queries = by_final.get("clean", 0)
    answered_queries = by_status.get("answered", 0)

    return {
        "preset": preset_name,
        "mode": "rule_judge" if use_judge else "rule_only",
        "weights": safe_json(WEIGHT_PRESETS[preset_name]),
        "total_queries": total,
        "attack_queries": len(attack_rows),
        "normal_queries": len(normal_rows),
        "answered_queries": answered_queries,
        "blocked_queries": by_status.get("blocked", 0),
        "error_queries": by_status.get("error", 0),
        "first_clean_queries": by_first.get("clean", 0),
        "first_suspicious_queries": by_first.get("suspicious", 0),
        "first_malicious_queries": by_first.get("malicious", 0),
        "final_clean_queries": final_clean_queries,
        "final_suspicious_queries": by_final.get("suspicious", 0),
        "final_malicious_queries": by_final.get("malicious", 0),
        "first_rule_clean_queries": by_first_rule.get("clean", 0),
        "first_rule_suspicious_queries": by_first_rule.get("suspicious", 0),
        "first_rule_malicious_queries": by_first_rule.get("malicious", 0),
        "final_rule_clean_queries": by_final_rule.get("clean", 0),
        "final_rule_suspicious_queries": by_final_rule.get("suspicious", 0),
        "final_rule_malicious_queries": by_final_rule.get("malicious", 0),
        "attack_query_success": attack_query_success,
        "attack_query_total": len(attack_rows),
        "attack_query_success_rate": round(attack_query_success / len(attack_rows), 4) if attack_rows else 0.0,
        "normal_false_positive_queries": safe_json(normal_fp_queries),
        "normal_false_positive_doc_ids": safe_json(sorted(set(normal_fp_docs))),
        "attack_docs_quarantined_during_normal_queries": safe_json(attack_docs_in_normal),
        "attack_query_misses": safe_json(attack_query_misses),
        "attack_doc_misses": safe_json(attack_doc_misses),
        "avg_rounds": round(avg_rounds, 4),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/eval_queries.csv")
    parser.add_argument("--output-dir", default="results/weight_experiments")
    parser.add_argument("--index-path", default="data/faiss_index")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--use-judge", action="store_true")
    parser.add_argument("--presets", nargs="*", default=list(WEIGHT_PRESETS.keys()))
    args = parser.parse_args()

    input_path = PROJECT_ROOT / args.input
    output_dir = PROJECT_ROOT / args.output_dir
    mode = "rule_judge" if args.use_judge else "rule_only"

    rows = read_queries(input_path, limit=args.limit)

    if not rows:
        raise SystemExit(f"No queries loaded from {input_path}")

    all_detail_rows = []
    summary_rows = []

    for preset_name in args.presets:
        if preset_name not in WEIGHT_PRESETS:
            raise SystemExit(f"Unknown preset: {preset_name}")

        detail_rows = evaluate_preset(
            preset_name=preset_name,
            rows=rows,
            k=args.k,
            max_rounds=args.max_rounds,
            index_path=args.index_path,
            use_judge=args.use_judge,
        )

        all_detail_rows.extend(detail_rows)
        summary = summarize_preset(preset_name, detail_rows, use_judge=args.use_judge)
        summary_rows.append(summary)

        print()
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    detail_path = output_dir / f"weight_detail_{mode}.csv"
    summary_path = output_dir / f"weight_summary_{mode}.csv"
    metrics_path = output_dir / f"weight_metrics_{mode}.json"

    write_csv(detail_path, all_detail_rows)
    write_csv(summary_path, summary_rows)

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 80)
    print(f"saved detail: {detail_path}")
    print(f"saved summary: {summary_path}")
    print(f"saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
