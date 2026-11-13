#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_EVAL_NORMALIZED_DIR = ROOT / "data" / "local_eval" / "normalized"


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"(?<=[。！？；])", text)
    return [p.strip() for p in parts if p.strip()]


def build_chunks_from_text(text: str, min_chars: int, max_chars: int) -> list[str]:
    sentences = split_sentences(text)
    chunks = []
    buf = ""

    for sent in sentences:
        if not buf:
            buf = sent
            continue

        if len(buf) + len(sent) <= max_chars:
            buf += sent
        else:
            if len(buf) >= min_chars:
                chunks.append(buf)
                buf = sent
            else:
                buf += sent

    if buf:
        if chunks and len(buf) < min_chars:
            chunks[-1] += buf
        else:
            chunks.append(buf)

    return chunks


def main() -> int:
    parser = argparse.ArgumentParser(description="Build chunks for local/held-out eval sections.")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--min-chars", type=int, default=250)
    parser.add_argument("--max-chars", type=int, default=750)
    args = parser.parse_args()

    in_path = LOCAL_EVAL_NORMALIZED_DIR / f"{args.dataset_id}.sections.jsonl"
    out_path = LOCAL_EVAL_NORMALIZED_DIR / f"{args.dataset_id}.chunks.jsonl"

    if not in_path.exists():
        raise FileNotFoundError(in_path)

    total = 0
    with in_path.open("r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
        for line in src:
            section = json.loads(line)
            section_chunks = build_chunks_from_text(
                section["text"],
                min_chars=args.min_chars,
                max_chars=args.max_chars,
            )

            kept_in_section = 0
            for chunk_text in section_chunks:
                # Drop accidental title-only chunks.
                if len(chunk_text.strip()) < args.min_chars:
                    continue

                kept_in_section += 1
                total += 1
                row = {
                    "dataset_id": args.dataset_id,
                    "title": section["title"],
                    "section_index": section["section_index"],
                    "section_title": section["title"],
                    "chunk_index": kept_in_section,
                    "global_chunk_index": total,
                    "char_count": len(chunk_text),
                    "text": chunk_text,
                    "usage": "local_eval_only",
                    "training": False,
                }
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved chunks: {out_path}")
    print(f"chunks: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
