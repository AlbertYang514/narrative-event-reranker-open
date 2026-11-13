#!/usr/bin/env python3
"""Evaluate a weak-label baseline on one or more soft-label JSONL files.

This is a data-agnostic diagnostic tool. Pass training or held-out files
explicitly; no private dataset paths are built into the code.
"""

import argparse
import json
from pathlib import Path

POSITIVE_STRATEGIES = {
    "adjacent_chunk",
    "same_section_gap_2",
    "same_section_gap_3",
}

NEGATIVE_STRATEGIES = {
    "random_distant_negative",
}


def load(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pairwise_auc(pos_scores, neg_scores):
    wins = 0
    ties = 0
    total = 0
    for p in pos_scores:
        for n in neg_scores:
            total += 1
            if p > n:
                wins += 1
            elif p == n:
                ties += 1
    if total == 0:
        return None
    return (wins + 0.5 * ties) / total


def eval_file(path: Path):
    rows = load(path)

    pos = [r for r in rows if r["strategy"] in POSITIVE_STRATEGIES]
    neg = [r for r in rows if r["strategy"] in NEGATIVE_STRATEGIES]
    if not pos or not neg:
        return rows, {}

    metrics = {}
    for field in ["rule_score", "teacher_expected_score", "final_score"]:
        pos_scores = [float(r[field]) for r in pos if r.get(field) is not None]
        neg_scores = [float(r[field]) for r in neg if r.get(field) is not None]
        if not pos_scores or not neg_scores:
            continue
        metrics[field] = {
            "pos_avg": sum(pos_scores) / len(pos_scores),
            "neg_avg": sum(neg_scores) / len(neg_scores),
            "auc": pairwise_auc(pos_scores, neg_scores),
        }

    return rows, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", help="Soft-label JSONL files")
    args = parser.parse_args()

    for raw_path in args.files:
        path = Path(raw_path)
        rows, metrics = eval_file(path)
        print("\n==", path, "==")
        print("rows:", len(rows))
        if not metrics:
            print("No positive/negative strategy mix found; skipping AUC.")
            continue
        for field, m in metrics.items():
            print(
                field,
                "pos_avg=", round(m["pos_avg"], 4),
                "neg_avg=", round(m["neg_avg"], 4),
                "gap=", round(m["pos_avg"] - m["neg_avg"], 4),
                "auc=", round(m["auc"], 4) if m["auc"] is not None else None,
            )

        print("top final_score:")
        for r in sorted(rows, key=lambda x: float(x["final_score"]), reverse=True)[:5]:
            print(" ", r["strategy"], r["pair_id"], r["final_score"])

        print("bottom final_score:")
        for r in sorted(rows, key=lambda x: float(x["final_score"]))[:5]:
            print(" ", r["strategy"], r["pair_id"], r["final_score"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
