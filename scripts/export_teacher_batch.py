#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data" / "datasets"


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a small teacher labeling batch from pair candidates.")
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--variant", default="simplified")
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--per-strategy", type=int, default=20)
    args = parser.parse_args()

    in_path = DATASET_DIR / f"{args.source_id}.{args.variant}.pair_candidates.jsonl"
    out_path = DATASET_DIR / f"{args.source_id}.{args.variant}.teacher_batch_001.jsonl"

    rows = [json.loads(line) for line in in_path.open("r", encoding="utf-8") if line.strip()]

    buckets = {}
    for row in rows:
        buckets.setdefault(row["strategy"], []).append(row)

    selected = []
    preferred = [
        "adjacent_chunk",
        "same_chapter_gap_2",
        "same_chapter_gap_3",
        "same_chapter_gap_4",
        "chapter_boundary",
        "random_distant_negative",
    ]

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

    print(f"Saved teacher batch: {out_path}")
    print(f"rows: {len(selected)}")
    for strategy in preferred:
        count = sum(1 for r in selected if r["strategy"] == strategy)
        if count:
            print(f"{strategy}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
