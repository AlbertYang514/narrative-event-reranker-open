#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NORMALIZED_DIR = ROOT / "data" / "normalized"
DATASET_DIR = ROOT / "data" / "datasets"


def load_chunks(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pair_id(source_id: str, a: dict, b: dict) -> str:
    return (
        f"{source_id}:"
        f"c{a['global_chunk_index']:05d}-"
        f"c{b['global_chunk_index']:05d}"
    )


def make_pair(source_id: str, a: dict, b: dict, strategy: str) -> dict:
    return {
        "task": "event_relevance",
        "pair_id": pair_id(source_id, a, b),
        "source_id": source_id,
        "strategy": strategy,
        "current": {
            "chapter_index": b["chapter_index"],
            "chapter_title": b["chapter_title"],
            "chunk_index": b["chunk_index"],
            "global_chunk_index": b["global_chunk_index"],
            "text": b["text"],
            "url": b.get("url"),
        },
        "candidate": {
            "chapter_index": a["chapter_index"],
            "chapter_title": a["chapter_title"],
            "chunk_index": a["chunk_index"],
            "global_chunk_index": a["global_chunk_index"],
            "text": a["text"],
            "url": a.get("url"),
        },
        "metadata": {
            "chapter_distance": b["chapter_index"] - a["chapter_index"],
            "chunk_distance": b["global_chunk_index"] - a["global_chunk_index"],
        },
        "label": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build chunk-pair candidates for teacher labeling.")
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--variant", default="simplified")
    parser.add_argument("--random-negatives", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--collection-mode",
        choices=["continuous", "short_story"],
        default="continuous",
        help="continuous keeps cross-chapter boundary pairs; short_story treats chapters as independent stories",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    chunks_path = NORMALIZED_DIR / f"{args.source_id}.{args.variant}.chunks.jsonl"
    out_path = DATASET_DIR / f"{args.source_id}.{args.variant}.pair_candidates.jsonl"

    chunks = load_chunks(chunks_path)
    by_global = {c["global_chunk_index"]: c for c in chunks}

    pairs = {}
    n = len(chunks)

    def add(a: dict, b: dict, strategy: str):
        if a["global_chunk_index"] == b["global_chunk_index"]:
            return
        if a["global_chunk_index"] > b["global_chunk_index"]:
            a, b = b, a
        row = make_pair(args.source_id, a, b, strategy)
        pairs[row["pair_id"]] = row

    # 1. Adjacent chunks.
    # In continuous novels, the last chunk of one chapter and the first chunk
    # of the next chapter may still be useful as a chapter_boundary pair below.
    # In short-story collections, chapters are independent stories, so adjacent
    # positives must stay within the same chapter/story.
    for i in range(n - 1):
        a = chunks[i]
        b = chunks[i + 1]
        if args.collection_mode == "short_story" and a["chapter_index"] != b["chapter_index"]:
            continue
        add(a, b, "adjacent_chunk")

    # 2. Same chapter near-distance chunks.
    for i, a in enumerate(chunks):
        for gap in (2, 3, 4):
            j = i + gap
            if j < n:
                b = chunks[j]
                if a["chapter_index"] == b["chapter_index"]:
                    add(a, b, f"same_chapter_gap_{gap}")

    # 3. Cross-chapter boundary pairs.
    # Only meaningful for continuous novels. For short-story collections,
    # cross-story boundaries are negatives, not weak positives.
    if args.collection_mode == "continuous":
        for i in range(n - 1):
            a = chunks[i]
            b = chunks[i + 1]
            if b["chapter_index"] == a["chapter_index"] + 1:
                add(a, b, "chapter_boundary")

    # 4. Random distant negatives.
    attempts = 0
    added_neg = 0
    while added_neg < args.random_negatives and attempts < args.random_negatives * 20:
        attempts += 1
        a, b = random.sample(chunks, 2)
        if a["global_chunk_index"] > b["global_chunk_index"]:
            a, b = b, a

        chunk_distance = b["global_chunk_index"] - a["global_chunk_index"]
        chapter_distance = b["chapter_index"] - a["chapter_index"]

        if chunk_distance < 20:
            continue
        if chapter_distance < 5:
            continue

        before = len(pairs)
        add(a, b, "random_distant_negative")
        if len(pairs) > before:
            added_neg += 1

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in pairs.values():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = {}
    for row in pairs.values():
        counts[row["strategy"]] = counts.get(row["strategy"], 0) + 1

    print(f"Saved pair candidates: {out_path}")
    print(f"total_pairs: {len(pairs)}")
    for k in sorted(counts):
        print(f"{k}: {counts[k]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
