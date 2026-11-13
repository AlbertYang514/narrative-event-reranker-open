#!/usr/bin/env python3
import argparse
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse, unquote, quote
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from http.client import IncompleteRead

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

USER_AGENT = "narrative-event-reranker/0.1 (+local public-domain research)"

DROP_LINK_PREFIXES = (
    "/wiki/Special:",
    "/wiki/Help:",
    "/wiki/Wikisource:",
    "/wiki/Category:",
    "/wiki/File:",
    "/wiki/Template:",
    "/wiki/Talk:",
    "/wiki/User:",
)

def safe_url(url: str) -> str:
    parsed = urlparse(url)
    path = quote(unquote(parsed.path), safe="/:%")
    query = quote(unquote(parsed.query), safe="=&?/:+,%")
    fragment = quote(unquote(parsed.fragment), safe="")
    return parsed._replace(path=path, query=query, fragment=fragment).geturl()


def fetch_html(url: str, delay: float = 1.0, retries: int = 3) -> str:
    safe = safe_url(url)
    last_error = None

    for attempt in range(1, retries + 1):
        req = Request(
            safe,
            headers={
                "User-Agent": USER_AGENT,
                "Accept-Encoding": "identity",
                "Connection": "close",
            },
        )
        try:
            with urlopen(req, timeout=60) as response:
                raw = response.read()
            time.sleep(delay)
            for enc in ("utf-8", "utf-8-sig", "gb18030", "big5"):
                try:
                    return raw.decode(enc)
                except UnicodeDecodeError:
                    pass
            return raw.decode("utf-8", errors="replace")
        except IncompleteRead as exc:
            last_error = exc
            print(f"WARN incomplete read attempt {attempt}/{retries}: {url}", file=sys.stderr)
            if exc.partial and len(exc.partial) > 2000:
                raw = exc.partial
                time.sleep(delay)
                return raw.decode("utf-8", errors="replace")
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
            print(f"WARN fetch failed attempt {attempt}/{retries}: {url}: {exc}", file=sys.stderr)

        time.sleep(delay * attempt)

    raise last_error if last_error else RuntimeError(f"failed to fetch: {url}")

def clean_title_from_url(url: str) -> str:
    path = unquote(urlparse(url).path)
    title = path.split("/wiki/", 1)[-1] if "/wiki/" in path else path.rsplit("/", 1)[-1]
    title = title.replace("_", " ").strip()
    return title

def get_content_root(soup: BeautifulSoup):
    return (
        soup.find(id="mw-content-text")
        or soup.find("main")
        or soup.find("article")
        or soup.body
        or soup
    )

def strip_wikisource_tail_noise(text: str) -> str:
    # Wikisource chapter pages often append license/footer text after "返回頁首".
    # For our raw training source, keep the work text and drop site/page chrome.
    text = re.split(r"\n?返回[頁页]首\n?", text, maxsplit=1)[0]

    # Extra defensive cuts for pages where the footer marker is absent.
    footer_patterns = [
        r"\n?此[^\n]{0,40}作品在全世界都属于.*\Z",
        r"\n?Public domain\s+Public domain\s+false\s+false\s*\Z",
    ]
    for pattern in footer_patterns:
        text = re.sub(pattern, "", text, flags=re.S)

    noise_lines = {
        "编辑",
        "編輯",
        "Public domain",
        "public domain",
        "false",
        "true",
        "返回頁首",
        "返回页首",
        "公有领域",
        "公有領域",
    }

    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped in noise_lines:
            continue
        if stripped.startswith("此清朝作品在全世界都属于"):
            continue
        if stripped.startswith("此作品在全世界都属于"):
            continue
        if "作者逝世已经超过100年" in stripped:
            continue
        lines.append(line)

    return "\n".join(lines)


def normalize_text(text: str) -> str:
    text = strip_wikisource_tail_noise(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip() + "\n"

def remove_noise(root):
    for tag in root.select(
        ".mw-editsection, .printfooter, .catlinks, .navbox, "
        ".metadata, .sistersitebox, table, script, style, noscript"
    ):
        tag.decompose()

def extract_page_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    root = get_content_root(soup)
    remove_noise(root)

    # MediaWiki pages often keep real text in parser-output.
    parser_output = root.select_one(".mw-parser-output")
    if parser_output:
        root = parser_output
        remove_noise(root)

    text = root.get_text("\n")
    return normalize_text(text)

def should_keep_chapter_link(href: str, base_url: str) -> bool:
    if not href:
        return False
    if href.startswith("#"):
        return False
    parsed_base = urlparse(base_url)
    full = urljoin(base_url, href)
    parsed = urlparse(full)

    if parsed.netloc and parsed.netloc != parsed_base.netloc:
        return False

    path = unquote(parsed.path)
    if not path.startswith("/wiki/"):
        return False

    for prefix in DROP_LINK_PREFIXES:
        if path.startswith(prefix):
            return False

    return True

def looks_like_chapter_title(text: str) -> bool:
    text = text.strip()
    if not text:
        return False

    # Common Wikisource chapter labels for late-Qing novels.
    if text in {"序", "自序", "自敘", "自叙", "楔子", "引子", "凡例"}:
        return True
    if re.fullmatch(r"《[^》]{1,40}》序", text):
        return True
    if re.fullmatch(r"第[一二三四五六七八九十百千〇零两]+回", text):
        return True
    if re.fullmatch(r"第\d{1,3}回", text):
        return True
    return False


def looks_like_chapter_path(path: str) -> bool:
    # Keep only normal /wiki/<work>/<chapter> pages.
    if not path.startswith("/wiki/"):
        return False
    rest = path.removeprefix("/wiki/")
    parts = rest.split("/")
    if len(parts) != 2:
        return False

    work, chapter = parts
    if not work or not chapter:
        return False

    # Exclude obvious non-work pages and sister-project pseudo paths.
    if ":" in work or ":" in chapter:
        return False

    chapter = chapter.strip()
    if chapter in {"序", "自序", "自敘", "自叙", "楔子", "引子", "凡例"}:
        return True
    if re.fullmatch(r"第\d{1,3}回", chapter):
        return True
    if re.fullmatch(r"\d{1,3}", chapter):
        return True
    if re.fullmatch(r"第[一二三四五六七八九十百千〇零两]+回", chapter):
        return True
    if re.fullmatch(r"《[^》]{1,40}》序", chapter):
        return True
    return False


def find_chapter_links(index_url: str, html: str, chapter_titles: list[str] | None = None) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    root = get_content_root(soup)
    parser_output = root.select_one(".mw-parser-output")
    if parser_output:
        root = parser_output

    title_whitelist = None
    if chapter_titles:
        title_whitelist = {normalize_text(str(x)).strip() for x in chapter_titles if normalize_text(str(x)).strip()}

    candidates = []
    seen = set()

    for a in root.find_all("a", href=True):
        href = a.get("href") or ""
        text = normalize_text(a.get_text(" ")).strip()
        if not text:
            continue
        if not should_keep_chapter_link(href, index_url):
            continue

        full_url = urljoin(index_url, href)
        parsed = urlparse(full_url)
        path = unquote(parsed.path).rstrip("/")

        if title_whitelist is not None:
            # For modern short-story collections such as Lu Xun's Nahan,
            # TOC entries may point to standalone pages like /wiki/狂人日記
            # rather than /wiki/吶喊/狂人日記. Keep exact TOC whitelist only.
            if text not in title_whitelist:
                continue
        else:
            # Wikisource often redirects simplified index titles to traditional
            # subpage paths, e.g. /wiki/老残游记 -> /wiki/老殘遊記/第01回.
            # Therefore do not require path.startswith(base_path + "/").
            # Instead, keep links that look like chapter pages both by link text
            # and by /wiki/<work>/<chapter> path shape.
            if not (looks_like_chapter_title(text) and looks_like_chapter_path(path)):
                continue

        key = parsed._replace(fragment="", query="").geturl()
        if key in seen:
            continue
        seen.add(key)

        candidates.append(
            {
                "title": text,
                "url": key,
            }
        )

    if title_whitelist is not None:
        order = {title: idx for idx, title in enumerate(chapter_titles or [])}
        candidates.sort(key=lambda item: order.get(item["title"], 10**9))

    return candidates

def load_manifest_row(manifest_path: Path, source_id: str) -> dict:
    with manifest_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("source_id") == source_id:
                return row
    raise ValueError(f"source_id not found in manifest: {source_id}")

def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch one Chinese Wikisource work by index page.")
    parser.add_argument("--manifest", required=True, help="manifest JSONL path")
    parser.add_argument("--source-id", required=True, help="source_id in manifest")
    parser.add_argument("--force", action="store_true", help="overwrite existing output")
    parser.add_argument("--delay", type=float, default=1.0, help="delay between requests")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    row = load_manifest_row(manifest_path, args.source_id)

    if row.get("allowed_for_training") is not True:
        print(f"Refusing source_id={args.source_id}: allowed_for_training is not true", file=sys.stderr)
        return 1
    if row.get("jurisdiction_checked") is not True:
        print(f"Refusing source_id={args.source_id}: jurisdiction_checked is not true", file=sys.stderr)
        return 1

    index_url = str(row.get("source_url") or "").strip()
    if "wikisource.org" not in urlparse(index_url).netloc:
        print("This script is only intended for Wikisource URLs.", file=sys.stderr)
        return 1

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_out = RAW_DIR / f"{args.source_id}.txt"
    chapters_out = PROCESSED_DIR / f"{args.source_id}.chapters.jsonl"
    meta_out = RAW_DIR / f"{args.source_id}.meta.json"

    if (raw_out.exists() or chapters_out.exists()) and not args.force:
        print("Output already exists. Use --force to overwrite.", file=sys.stderr)
        print(raw_out)
        print(chapters_out)
        return 1

    print(f"Fetching index: {index_url}")
    index_html = fetch_html(index_url, delay=args.delay)
    chapter_links = find_chapter_links(index_url, index_html, row.get("chapter_titles"))

    if not chapter_links:
        print("No chapter subpage links found. The work may be a single-page text.", file=sys.stderr)
        text = extract_page_text(index_html)
        raw_out.write_text(text, encoding="utf-8")
        chapters_out.write_text(
            json.dumps(
                {
                    "source_id": args.source_id,
                    "chapter_index": 1,
                    "title": row.get("title") or clean_title_from_url(index_url),
                    "url": index_url,
                    "char_count": len(text),
                    "text": text,
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        meta_out.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Saved single-page text: {raw_out}")
        return 0

    print(f"Found {len(chapter_links)} chapter links.")

    all_parts = []
    chapter_rows = []

    for idx, chapter in enumerate(chapter_links, start=1):
        title = chapter["title"]
        url = chapter["url"]
        print(f"[{idx}/{len(chapter_links)}] {title} -> {url}")

        try:
            html = fetch_html(url, delay=args.delay)
            text = extract_page_text(html)
        except (HTTPError, URLError, OSError) as exc:
            print(f"WARN failed chapter: {url}: {exc}", file=sys.stderr)
            continue

        # Drop extremely short accidental pages.
        if len(text) < 100:
            print(f"WARN short chapter skipped: {title} ({len(text)} chars)", file=sys.stderr)
            continue

        all_parts.append(f"\n\n# {title}\n\n{text.strip()}\n")
        chapter_rows.append(
            {
                "source_id": args.source_id,
                "chapter_index": len(chapter_rows) + 1,
                "title": title,
                "url": url,
                "char_count": len(text),
                "text": text,
            }
        )

    if not chapter_rows:
        print("No usable chapters fetched.", file=sys.stderr)
        return 1

    full_text = normalize_text("\n".join(all_parts))
    raw_out.write_text(full_text, encoding="utf-8")

    with chapters_out.open("w", encoding="utf-8") as f:
        for chapter in chapter_rows:
            f.write(json.dumps(chapter, ensure_ascii=False) + "\n")

    meta = {
        **row,
        "chapter_count": len(chapter_rows),
        "char_count": len(full_text),
        "index_url": index_url,
    }
    meta_out.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("")
    print(f"Saved raw text: {raw_out}")
    print(f"Saved chapters: {chapters_out}")
    print(f"Saved metadata: {meta_out}")
    print(f"Chapters: {len(chapter_rows)}")
    print(f"Characters: {len(full_text)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
