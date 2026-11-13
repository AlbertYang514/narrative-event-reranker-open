#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

def load_manifest(path: Path) -> dict[str, dict]:
    sources = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            source_id = str(row.get("source_id") or "").strip()
            if not source_id:
                raise ValueError(f"{path}:{line_no}: missing source_id")
            sources[source_id] = row
    return sources

def fetch_url(url: str) -> str:
    request = Request(
        url,
        headers={
            "User-Agent": "narrative-event-reranker/0.1 (+local research; contact: local)"
        },
    )
    with urlopen(request, timeout=30) as response:
        raw = response.read()

    for encoding in ("utf-8", "utf-8-sig", "gb18030", "big5"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue

    return raw.decode("utf-8", errors="replace")

def read_file_url(url: str) -> str:
    path = Path(url.removeprefix("file://")).expanduser()
    return path.read_text(encoding="utf-8")

def main() -> int:
    parser = argparse.ArgumentParser(description="Import one approved text source into data/raw.")
    parser.add_argument("--manifest", required=True, help="Path to source manifest JSONL.")
    parser.add_argument("--source-id", required=True, help="source_id from manifest.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing raw text.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    sources = load_manifest(manifest_path)

    source = sources.get(args.source_id)
    if not source:
        print(f"source_id not found in manifest: {args.source_id}", file=sys.stderr)
        return 1

    if source.get("allowed_for_training") is not True:
        print(f"source_id is not allowed_for_training=true: {args.source_id}", file=sys.stderr)
        return 1

    if source.get("jurisdiction_checked") is not True:
        print(f"source_id requires jurisdiction_checked=true: {args.source_id}", file=sys.stderr)
        return 1

    url = str(source.get("source_url") or "").strip()
    if not url:
        print("source_url is empty", file=sys.stderr)
        return 1

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"{args.source_id}.txt"

    if out_path.exists() and not args.force:
        print(f"Refusing to overwrite existing file: {out_path}", file=sys.stderr)
        return 1

    try:
        if url.startswith("file://"):
            text = read_file_url(url)
        elif url.startswith("http://") or url.startswith("https://"):
            text = fetch_url(url)
        else:
            print("source_url must start with http(s):// or file://", file=sys.stderr)
            return 1
    except (URLError, HTTPError, OSError) as exc:
        print(f"Failed to import source: {exc}", file=sys.stderr)
        return 1

    text = text.replace("\r\n", "\n").replace("\r", "\n").strip() + "\n"
    out_path.write_text(text, encoding="utf-8")

    print(f"Imported {args.source_id}")
    print(f"Title: {source.get('title')}")
    print(f"Raw text: {out_path}")
    print(f"Characters: {len(text)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
