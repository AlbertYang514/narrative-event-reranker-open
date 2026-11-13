#!/usr/bin/env python3
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOFT_LABEL_DIR = ROOT / "data" / "soft_labels" / "train"
DEFAULT_OUT_DIR = ROOT / "data" / "train_splits" / "rule_only_v1"


def load_rows(input_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(input_dir.glob("*.soft_labels.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def split_group(rows: list[dict], rng: random.Random) -> tuple[list[dict], list[dict], list[dict]]:
    rows = list(rows)
    rng.shuffle(rows)

    n = len(rows)
    if n == 1:
        return rows, [], []
    if n == 2:
        return rows[:1], rows[1:], []

    n_train = int(n * 0.8)
    n_valid = int(n * 0.1)

    # Keep at least one validation and one test item for groups with enough rows.
    if n >= 10:
        n_valid = max(1, n_valid)
        n_test = max(1, n - n_train - n_valid)
        if n_train + n_valid + n_test > n:
            n_train = n - n_valid - n_test
    else:
        n_valid = max(1, n_valid)
        n_test = n - n_train - n_valid
        if n_test < 0:
            n_test = 0
            n_train = n - n_valid

    train = rows[:n_train]
    valid = rows[n_train:n_train + n_valid]
    test = rows[n_train + n_valid:]
    return train, valid, test


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(name: str, rows: list[dict]) -> None:
    by_source = defaultdict(int)
    by_strategy = defaultdict(int)
    for r in rows:
        by_source[r.get("source_id")] += 1
        by_strategy[r.get("strategy")] += 1

    print(f"\n=== {name} ===")
    print(f"rows: {len(rows)}")
    print("sources:")
    for k in sorted(by_source):
        print(f"  {k}: {by_source[k]}")
    print("strategies:")
    for k in sorted(by_strategy):
        print(f"  {k}: {by_strategy[k]}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build stratified train/valid/test splits from rule-only soft labels.")
    parser.add_argument("--input-dir", default=str(SOFT_LABEL_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    rng = random.Random(args.seed)

    rows = load_rows(input_dir)
    if not rows:
        raise SystemExit(f"No soft label rows found in {input_dir}")

    groups = defaultdict(list)
    for row in rows:
        key = (row.get("source_id"), row.get("strategy"))
        groups[key].append(row)

    train, valid, test = [], [], []

    for key in sorted(groups):
        g_train, g_valid, g_test = split_group(groups[key], rng)
        train.extend(g_train)
        valid.extend(g_valid)
        test.extend(g_test)

    rng.shuffle(train)
    rng.shuffle(valid)
    rng.shuffle(test)

    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "valid.jsonl", valid)
    write_jsonl(out_dir / "test.jsonl", test)

    summarize("train", train)
    summarize("valid", valid)
    summarize("test", test)

    all_ids = [r.get("pair_id") for r in train + valid + test]
    if len(all_ids) != len(set(all_ids)):
        raise SystemExit("ERROR: duplicate pair_id across splits")

    print(f"\nWrote splits to {out_dir}")
    print(f"total: {len(train) + len(valid) + len(test)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
