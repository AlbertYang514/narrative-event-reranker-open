#!/usr/bin/env python3
import json
import sys
from pathlib import Path

REQUIRED = {
    "source_id",
    "title",
    "author",
    "source_url",
    "license",
    "jurisdiction_checked",
    "allowed_for_training",
}

def validate_file(path: Path) -> int:
    errors = 0
    seen_ids = set()

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"{path}:{line_no}: invalid JSON: {exc}")
                errors += 1
                continue

            missing = REQUIRED - set(row)
            if missing:
                print(f"{path}:{line_no}: missing fields: {sorted(missing)}")
                errors += 1

            source_id = str(row.get("source_id") or "").strip()
            if not source_id:
                print(f"{path}:{line_no}: empty source_id")
                errors += 1
            elif source_id in seen_ids:
                print(f"{path}:{line_no}: duplicate source_id: {source_id}")
                errors += 1
            seen_ids.add(source_id)

            if not isinstance(row.get("jurisdiction_checked"), bool):
                print(f"{path}:{line_no}: jurisdiction_checked must be boolean")
                errors += 1

            if not isinstance(row.get("allowed_for_training"), bool):
                print(f"{path}:{line_no}: allowed_for_training must be boolean")
                errors += 1

            if row.get("allowed_for_training") is True and row.get("jurisdiction_checked") is not True:
                print(f"{path}:{line_no}: allowed_for_training=true requires jurisdiction_checked=true")
                errors += 1

            license_value = str(row.get("license") or "").strip().lower()
            if not license_value:
                print(f"{path}:{line_no}: empty license")
                errors += 1

            url = str(row.get("source_url") or "").strip()
            if url and not (url.startswith("http://") or url.startswith("https://") or url.startswith("file://")):
                print(f"{path}:{line_no}: source_url should be http(s):// or file://")
                errors += 1

    return errors

def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: scripts/validate_manifest.py <manifest.jsonl> [...]")
        return 2

    total_errors = 0
    for arg in sys.argv[1:]:
        total_errors += validate_file(Path(arg))

    if total_errors:
        print(f"Manifest validation failed: {total_errors} error(s)")
        return 1

    print("Manifest validation OK")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
