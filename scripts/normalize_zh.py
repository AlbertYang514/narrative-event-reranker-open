#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from opencc import OpenCC

ROOT = Path(__file__).resolve().parents[1]
NORMALIZED_DIR = ROOT / "data" / "normalized"


def convert_text(text: str, mode: str) -> str:
    converter = OpenCC(mode)
    return converter.convert(text)


def normalize_raw(source_id: str, mode: str = "t2s") -> None:
    raw_path = ROOT / "data" / "raw" / f"{source_id}.txt"
    chapters_path = ROOT / "data" / "processed" / f"{source_id}.chapters.jsonl"

    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    if not chapters_path.exists():
        raise FileNotFoundError(chapters_path)

    NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)

    suffix = "simplified" if mode == "t2s" else mode
    out_text_path = NORMALIZED_DIR / f"{source_id}.{suffix}.txt"
    out_chapters_path = NORMALIZED_DIR / f"{source_id}.{suffix}.chapters.jsonl"

    raw_text = raw_path.read_text(encoding="utf-8")
    out_text_path.write_text(convert_text(raw_text, mode), encoding="utf-8")

    with chapters_path.open("r", encoding="utf-8") as src, out_chapters_path.open("w", encoding="utf-8") as dst:
        for line in src:
            row = json.loads(line)
            row["title_original"] = row.get("title")
            row["text_original_char_count"] = row.get("char_count")
            row["title"] = convert_text(row["title"], mode)
            row["text"] = convert_text(row["text"], mode)
            row["char_count"] = len(row["text"])
            row["normalization"] = mode
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved normalized text: {out_text_path}")
    print(f"Saved normalized chapters: {out_chapters_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize Chinese text variants without overwriting raw data.")
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--mode", default="t2s", help="OpenCC mode, default t2s")
    args = parser.parse_args()

    normalize_raw(args.source_id, args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
