from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"Invalid JSON in {path}:{line_no}: {e}") from e


def get_score(row: dict) -> float:
    for key in ("final_score", "score", "label", "teacher_expected_score"):
        if key in row:
            return float(row[key])
    raise KeyError(f"No score field found. Available keys: {sorted(row.keys())}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-dir",
        default="data/train_splits/rule_only_v1",
        help="Directory containing train.jsonl / valid.jsonl / test.jsonl",
    )
    parser.add_argument("--examples", type=int, default=3)
    args = parser.parse_args()

    split_dir = Path(args.split_dir)

    total = 0
    all_pair_ids: dict[str, str] = {}
    overlap = defaultdict(list)

    for split_name in ("train", "valid", "test"):
        path = split_dir / f"{split_name}.jsonl"
        if not path.exists():
            raise SystemExit(f"Missing split file: {path}")

        rows = list(load_jsonl(path))
        total += len(rows)

        source_counts = Counter()
        strategy_counts = Counter()
        training_counts = Counter()
        scores = []
        missing_text = 0

        print(f"\n=== {split_name} ===")
        print(f"rows: {len(rows)}")

        for row in rows:
            source_id = row.get("source_id", "<missing>")
            pair_id = row.get("pair_id", "<missing>")
            strategy = row.get("strategy", row.get("label_strategy", "<missing>"))
            training = row.get("training", "<missing>")

            source_counts[source_id] += 1
            strategy_counts[strategy] += 1
            training_counts[str(training)] += 1
            scores.append(get_score(row))

            current_text = row.get("current_text", "")
            candidate_text = row.get("candidate_text", "")
            if not current_text or not candidate_text:
                missing_text += 1

            if pair_id in all_pair_ids:
                overlap[pair_id].extend([all_pair_ids[pair_id], split_name])
            else:
                all_pair_ids[pair_id] = split_name

        print(f"sources: {dict(source_counts)}")
        print(f"training flags: {dict(training_counts)}")
        print(f"score min/max/mean: {min(scores):.4f} / {max(scores):.4f} / {mean(scores):.4f}")
        print(f"missing current_text/candidate_text: {missing_text}")

        print("strategies:")
        for k, v in strategy_counts.most_common():
            print(f"  {k}: {v}")

        if args.examples > 0:
            print("\nexamples:")
            for row in rows[: args.examples]:
                print(
                    json.dumps(
                        {
                            "source_id": row.get("source_id"),
                            "pair_id": row.get("pair_id"),
                            "strategy": row.get("strategy", row.get("label_strategy")),
                            "score": get_score(row),
                            "training": row.get("training"),
                            "current_text_preview": row.get("current_text", "")[:80],
                            "candidate_text_preview": row.get("candidate_text", "")[:80],
                        },
                        ensure_ascii=False,
                    )
                )

    print("\n=== total ===")
    print(f"rows: {total}")
    print(f"unique pair_ids: {len(all_pair_ids)}")
    print(f"overlapping pair_ids across splits: {len(overlap)}")

    if overlap:
        print("overlap examples:")
        for pair_id, splits in list(overlap.items())[:10]:
            print(f"  {pair_id}: {sorted(set(splits))}")
        raise SystemExit("Split leakage detected: pair_id appears in multiple splits.")

    print("OK: no pair_id overlap across train/valid/test.")


if __name__ == "__main__":
    main()
