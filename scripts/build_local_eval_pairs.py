#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_EVAL_NORMALIZED_DIR = ROOT / "data" / "local_eval" / "normalized"
LOCAL_EVAL_DATASET_DIR = ROOT / "data" / "local_eval" / "datasets"


def load_chunks(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open("r", encoding="utf-8") if line.strip()]


def pair_id(dataset_id: str, a: dict, b: dict) -> str:
    return f"{dataset_id}:c{a['global_chunk_index']:05d}-c{b['global_chunk_index']:05d}"


def make_pair(dataset_id: str, a: dict, b: dict, strategy: str) -> dict:
    return {
        "task": "event_relevance",
        "pair_id": pair_id(dataset_id, a, b),
        "dataset_id": dataset_id,
        "source_type": "local_eval",
        "usage": "local_eval_only",
        "training": False,
        "strategy": strategy,
        "current": {
            "section_index": b["section_index"],
            "section_title": b["section_title"],
            "chunk_index": b["chunk_index"],
            "global_chunk_index": b["global_chunk_index"],
            "text": b["text"],
        },
        "candidate": {
            "section_index": a["section_index"],
            "section_title": a["section_title"],
            "chunk_index": a["chunk_index"],
            "global_chunk_index": a["global_chunk_index"],
            "text": a["text"],
        },
        "metadata": {
            "section_distance": b["section_index"] - a["section_index"],
            "chunk_distance": b["global_chunk_index"] - a["global_chunk_index"],
        },
        "label": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build local/held-out eval chunk-pair candidates.")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--random-negatives", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    in_path = LOCAL_EVAL_NORMALIZED_DIR / f"{args.dataset_id}.chunks.jsonl"
    out_path = LOCAL_EVAL_DATASET_DIR / f"{args.dataset_id}.pair_candidates.jsonl"

    chunks = load_chunks(in_path)
    pairs = {}
    n = len(chunks)

    def add(a: dict, b: dict, strategy: str):
        if a["global_chunk_index"] == b["global_chunk_index"]:
            return
        if a["global_chunk_index"] > b["global_chunk_index"]:
            a, b = b, a
        row = make_pair(args.dataset_id, a, b, strategy)
        pairs[row["pair_id"]] = row

    for i in range(n - 1):
        add(chunks[i], chunks[i + 1], "adjacent_chunk")

    for i, a in enumerate(chunks):
        for gap in (2, 3, 4):
            j = i + gap
            if j < n:
                b = chunks[j]
                if a["section_index"] == b["section_index"]:
                    add(a, b, f"same_section_gap_{gap}")

    for i in range(n - 1):
        a = chunks[i]
        b = chunks[i + 1]
        if b["section_index"] == a["section_index"] + 1:
            add(a, b, "section_boundary")

    attempts = 0
    added_neg = 0
    while added_neg < args.random_negatives and attempts < args.random_negatives * 30:
        attempts += 1
        a, b = random.sample(chunks, 2)
        if a["global_chunk_index"] > b["global_chunk_index"]:
            a, b = b, a

        chunk_distance = b["global_chunk_index"] - a["global_chunk_index"]
        section_distance = b["section_index"] - a["section_index"]

        if chunk_distance < 20:
            continue
        if section_distance < 3:
            continue

        before = len(pairs)
        add(a, b, "random_distant_negative")
        if len(pairs) > before:
            added_neg += 1

    LOCAL_EVAL_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in pairs.values():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = {}
    for row in pairs.values():
        counts[row["strategy"]] = counts.get(row["strategy"], 0) + 1

    print(f"Saved local/held-out eval pairs: {out_path}")
    print(f"total_pairs: {len(pairs)}")
    for k in sorted(counts):
        print(f"{k}: {counts[k]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
