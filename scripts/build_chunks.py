#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NORMALIZED_DIR = ROOT / "data" / "normalized"


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
    parser = argparse.ArgumentParser(description="Build paragraph-like chunks from normalized chapter JSONL.")
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--variant", default="simplified")
    parser.add_argument("--min-chars", type=int, default=300)
    parser.add_argument("--max-chars", type=int, default=900)
    args = parser.parse_args()

    in_path = NORMALIZED_DIR / f"{args.source_id}.{args.variant}.chapters.jsonl"
    out_path = NORMALIZED_DIR / f"{args.source_id}.{args.variant}.chunks.jsonl"

    if not in_path.exists():
        raise FileNotFoundError(in_path)

    total = 0
    with in_path.open("r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
        for line in src:
            chapter = json.loads(line)
            chapter_chunks = build_chunks_from_text(
                chapter["text"],
                min_chars=args.min_chars,
                max_chars=args.max_chars,
            )

            for i, chunk_text in enumerate(chapter_chunks, start=1):
                total += 1
                row = {
                    "source_id": args.source_id,
                    "variant": args.variant,
                    "chapter_index": chapter["chapter_index"],
                    "chapter_title": chapter["title"],
                    "chunk_index": i,
                    "global_chunk_index": total,
                    "char_count": len(chunk_text),
                    "text": chunk_text,
                    "url": chapter.get("url"),
                }
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved chunks: {out_path}")
    print(f"chunks: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
