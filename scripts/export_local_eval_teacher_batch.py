#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_EVAL_DATASET_DIR = ROOT / "data" / "local_eval" / "datasets"


def main() -> int:
    parser = argparse.ArgumentParser(description="Export local/held-out eval teacher labeling batch.")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--per-strategy", type=int, default=10)
    args = parser.parse_args()

    in_path = LOCAL_EVAL_DATASET_DIR / f"{args.dataset_id}.pair_candidates.jsonl"
    out_path = LOCAL_EVAL_DATASET_DIR / f"{args.dataset_id}.teacher_batch_001.jsonl"

    rows = [json.loads(line) for line in in_path.open("r", encoding="utf-8") if line.strip()]

    buckets = {}
    for row in rows:
        if row.get("training") is not False or row.get("usage") != "local_eval_only":
            raise SystemExit(f"Refusing non-local-eval row: {row.get('pair_id')}")
        buckets.setdefault(row["strategy"], []).append(row)

    preferred = [
        "adjacent_chunk",
        "same_section_gap_2",
        "same_section_gap_3",
        "same_section_gap_4",
        "section_boundary",
        "random_distant_negative",
    ]

    selected = []
    for strategy in preferred:
        for row in buckets.get(strategy, [])[:args.per_strategy]:
            selected.append(row)
            if len(selected) >= args.limit:
                break
        if len(selected) >= args.limit:
            break

    with out_path.open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved local/held-out eval teacher batch: {out_path}")
    print(f"rows: {len(selected)}")
    for strategy in preferred:
        count = sum(1 for r in selected if r["strategy"] == strategy)
        if count:
            print(f"{strategy}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
