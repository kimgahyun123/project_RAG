# scripts/make_eval_queries.py
from __future__ import annotations

import argparse
import csv
from pathlib import Path


ATTACK_QUERY_IDS = {1, 22, 33}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/eval_queries.csv")
    parser.add_argument("--normal-limit", type=int, default=30)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except Exception as exc:
        raise ImportError("datasets package is required: pip install datasets") from exc

    ds = load_dataset("isaacus/legal-rag-bench", "qa")
    split = ds["test"] if "test" in ds else next(iter(ds.values()))

    rows = []

    for idx, item in enumerate(split, start=1):
        question = (
            item.get("question")
            or item.get("query")
            or item.get("input")
            or item.get("prompt")
            or ""
        )
        question = str(question).strip()
        if not question:
            continue

        if idx in ATTACK_QUERY_IDS:
            rows.append({
                "query_id": idx,
                "kind": "attack",
                "query": question,
            })

    normal_count = 0
    for idx, item in enumerate(split, start=1):
        if idx in ATTACK_QUERY_IDS:
            continue

        question = (
            item.get("question")
            or item.get("query")
            or item.get("input")
            or item.get("prompt")
            or ""
        )
        question = str(question).strip()
        if not question:
            continue

        rows.append({
            "query_id": idx,
            "kind": "normal",
            "query": question,
        })
        normal_count += 1
        if normal_count >= args.normal_limit:
            break

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "kind", "query"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved: {out_path}")
    print(f"attack: {sum(1 for r in rows if r['kind'] == 'attack')}")
    print(f"normal: {sum(1 for r in rows if r['kind'] == 'normal')}")


if __name__ == "__main__":
    main()
