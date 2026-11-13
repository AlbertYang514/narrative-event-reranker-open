#!/usr/bin/env python3
"""Report soft-label statistics for one or more JSONL files or directories."""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def load(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def collect_files(args) -> list[Path]:
    files: list[Path] = []
    for raw in args:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.glob("*.soft_labels.jsonl")))
        else:
            files.append(path)
    if not files:
        raise SystemExit("No soft-label files found")
    return files


def report(path: Path) -> None:
    rows = load(path)
    print("\n==", path, "==")
    print("rows:", len(rows))
    print("training:", Counter(r.get("training") for r in rows))
    print("usage:", Counter(r.get("usage") for r in rows))
    print("strategies:", Counter(r.get("strategy") for r in rows))

    scores = [float(r["final_score"]) for r in rows if r.get("final_score") is not None]
    if scores:
        print("final_score:", {
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
            "avg": round(statistics.mean(scores), 4),
        })

    by_strategy = defaultdict(list)
    for r in rows:
        by_strategy[r.get("strategy")].append(r)

    print("strategy means:")
    for strategy, items in sorted(by_strategy.items(), key=lambda item: str(item[0])):
        vals = [float(r["final_score"]) for r in items if r.get("final_score") is not None]
        teacher_vals = [
            float(r["teacher_expected_score"])
            for r in items
            if r.get("teacher_expected_score") is not None
        ]
        if not vals:
            continue
        print(
            " ",
            strategy,
            "n=", len(items),
            "final_avg=", round(statistics.mean(vals), 4),
            "teacher_avg=", round(statistics.mean(teacher_vals), 4) if teacher_vals else None,
        )

    leaked = [
        r for r in rows
        if r.get("usage") == "local_eval_only" and r.get("training") is not False
    ]
    if leaked:
        raise SystemExit(f"HELD-OUT TRAINING LEAK: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Soft-label JSONL files or directories")
    args = parser.parse_args()

    for path in collect_files(args.paths):
        report(path)

    print("\nOK: soft-label reports complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
