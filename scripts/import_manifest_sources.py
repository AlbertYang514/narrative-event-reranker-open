#!/usr/bin/env python3
import argparse
import json
import re
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

REQUIRED = {
    "source_id",
    "title",
    "author",
    "source_url",
    "license",
    "jurisdiction_checked",
    "allowed_for_training",
}

def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc

            missing = REQUIRED - set(row)
            if missing:
                raise ValueError(f"{path}:{line_no}: missing fields: {sorted(missing)}")

            rows.append(row)
    return rows

def fetch_bytes(url: str, timeout: int = 45) -> bytes:
    req = Request(
        url,
        headers={
            "User-Agent": "narrative-event-reranker/0.1 (+local research; public-domain import)"
        },
    )
    with urlopen(req, timeout=timeout) as response:
        return response.read()

def decode_bytes(raw: bytes) -> str:
    for enc in ("utf-8", "utf-8-sig", "gb18030", "big5"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace")

def html_to_text(html: str) -> str:
    if BeautifulSoup is None:
        # crude fallback if bs4 is not installed
        text = re.sub(r"(?is)<script.*?>.*?</script>", "\n", html)
        text = re.sub(r"(?is)<style.*?>.*?</style>", "\n", text)
        text = re.sub(r"(?s)<[^>]+>", "\n", text)
        return text

    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # Prefer main content when obvious.
    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find(id="mw-content-text")  # Wikisource / MediaWiki
        or soup.body
        or soup
    )
    return main.get_text("\n")

def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    return text.strip() + "\n"

def looks_html(url: str, text: str) -> bool:
    path = urlparse(url).path.lower()
    if path.endswith((".html", ".htm")):
        return True
    head = text[:1000].lower()
    return "<html" in head or "<!doctype html" in head or "<body" in head

def import_source(row: dict, force: bool = False, delay: float = 1.0) -> bool:
    source_id = str(row["source_id"]).strip()
    url = str(row["source_url"]).strip()
    out_path = RAW_DIR / f"{source_id}.txt"
    meta_path = RAW_DIR / f"{source_id}.meta.json"

    if row.get("allowed_for_training") is not True:
        print(f"SKIP {source_id}: allowed_for_training is not true")
        return False

    if row.get("jurisdiction_checked") is not True:
        print(f"SKIP {source_id}: jurisdiction_checked is not true")
        return False

    if out_path.exists() and not force:
        print(f"SKIP {source_id}: already exists")
        return False

    print(f"IMPORT {source_id}: {row.get('title')}")

    if url.startswith("file://"):
        source_path = Path(url.removeprefix("file://")).expanduser()
        text = source_path.read_text(encoding="utf-8")
    elif url.startswith("http://") or url.startswith("https://"):
        raw = fetch_bytes(url)
        text = decode_bytes(raw)
        if looks_html(url, text):
            text = html_to_text(text)
        time.sleep(delay)
    else:
        raise ValueError(f"{source_id}: unsupported source_url: {url}")

    text = normalize_text(text)
    if len(text) < 500:
        print(f"WARN {source_id}: imported text is very short ({len(text)} chars)")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    meta_path.write_text(
        json.dumps(row, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"OK {source_id}: {len(text)} chars -> {out_path}")
    return True

def main() -> int:
    parser = argparse.ArgumentParser(description="Import approved text sources from manifest JSONL.")
    parser.add_argument("--manifest", required=True, help="manifest JSONL path")
    parser.add_argument("--source-id", help="import only one source_id")
    parser.add_argument("--force", action="store_true", help="overwrite existing raw text")
    parser.add_argument("--delay", type=float, default=1.0, help="delay between HTTP requests")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    rows = read_jsonl(manifest_path)

    if args.source_id:
        rows = [row for row in rows if row.get("source_id") == args.source_id]
        if not rows:
            print(f"source_id not found: {args.source_id}", file=sys.stderr)
            return 1

    imported = 0
    failed = 0

    for row in rows:
        try:
            if import_source(row, force=args.force, delay=args.delay):
                imported += 1
        except (OSError, HTTPError, URLError, ValueError) as exc:
            failed += 1
            print(f"FAIL {row.get('source_id')}: {exc}", file=sys.stderr)

    print(f"Done. imported={imported} failed={failed}")
    return 1 if failed else 0

if __name__ == "__main__":
    raise SystemExit(main())
