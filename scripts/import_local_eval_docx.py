#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

from docx import Document

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "local_eval" / "raw"
NORMALIZED_DIR = ROOT / "data" / "local_eval" / "normalized"


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip() + "\n"


def read_docx(path: Path) -> str:
    doc = Document(str(path))
    parts = []
    for p in doc.paragraphs:
        text = p.text.strip()
        if text:
            parts.append(text)
    return normalize_text("\n\n".join(parts))


def split_sections(text: str) -> list[dict]:
    # 保守切法：按明显标题/空行结构粗切；如果没有标题，就后续 chunk 阶段再切。
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    sections = []
    current_title = "正文"
    buf = []

    title_pat = re.compile(r"^(第[一二三四五六七八九十百零〇0-9]+[章节卷回]|[一二三四五六七八九十百零〇0-9]+、|番外|楔子|尾声|后记|後記).{0,30}$")

    def flush():
        nonlocal buf, current_title
        if buf:
            body = "\n".join(buf).strip() + "\n"
            sections.append({
                "section_index": len(sections) + 1,
                "title": current_title,
                "char_count": len(body),
                "text": body,
            })
            buf = []

    for para in paras:
        if title_pat.match(para) and len(buf) > 0:
            flush()
            current_title = para
        elif title_pat.match(para) and not buf:
            current_title = para
        else:
            buf.append(para)

    flush()
    return sections


def main() -> int:
    parser = argparse.ArgumentParser(description="Import local/held-out eval DOCX into raw text and section JSONL.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset-id", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    row = None
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item["dataset_id"] == args.dataset_id:
            row = item
            break

    if row is None:
        raise SystemExit(f"dataset_id not found: {args.dataset_id}")

    if row.get("training") is not False or row.get("usage") != "local_eval_only":
        raise SystemExit("Refusing: local/held-out eval sources must have training=false and usage=local_eval_only")

    source_path = Path(row["source_path"])
    if not source_path.exists():
        raise FileNotFoundError(source_path)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)

    text = read_docx(source_path)
    sections = split_sections(text)

    raw_out = RAW_DIR / f"{args.dataset_id}.txt"
    sections_out = NORMALIZED_DIR / f"{args.dataset_id}.sections.jsonl"
    meta_out = RAW_DIR / f"{args.dataset_id}.meta.json"

    raw_out.write_text(text, encoding="utf-8")
    with sections_out.open("w", encoding="utf-8") as f:
        for section in sections:
            section = {
                "dataset_id": args.dataset_id,
                "title": row["title"],
                **section,
            }
            f.write(json.dumps(section, ensure_ascii=False) + "\n")

    meta = {
        **row,
        "char_count": len(text),
        "section_count": len(sections),
    }
    meta_out.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Saved raw text: {raw_out}")
    print(f"Saved sections: {sections_out}")
    print(f"Saved metadata: {meta_out}")
    print(f"sections: {len(sections)}")
    print(f"characters: {len(text)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
