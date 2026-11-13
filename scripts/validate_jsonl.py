#!/usr/bin/env python3
import json
import sys
from pathlib import Path

REQUIRED = {
    "task",
    "source_id",
    "current_text",
    "candidate_event",
    "metadata",
    "label",
}

ALLOWED_LABELS = {
    "not_related",
    "weak_related",
    "strong_related",
    "same_event",
    "uncertain",
}

def validate_file(path: Path) -> int:
    errors = 0
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

            if row.get("label") not in ALLOWED_LABELS:
                print(f"{path}:{line_no}: invalid label: {row.get('label')!r}")
                errors += 1

            current = str(row.get("current_text") or "")
            candidate = str(row.get("candidate_event") or "")
            ev_current = str(row.get("evidence_current") or "")
            ev_candidate = str(row.get("evidence_candidate") or "")

            if ev_current and ev_current not in current:
                print(f"{path}:{line_no}: evidence_current not found in current_text")
                errors += 1

            if ev_candidate and ev_candidate not in candidate:
                print(f"{path}:{line_no}: evidence_candidate not found in candidate_event")
                errors += 1

    return errors

def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: scripts/validate_jsonl.py <file.jsonl> [...]")
        return 2
    total_errors = 0
    for arg in sys.argv[1:]:
        total_errors += validate_file(Path(arg))
    if total_errors:
        print(f"Validation failed: {total_errors} error(s)")
        return 1
    print("Validation OK")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
