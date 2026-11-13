#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


DEFAULT_THRESHOLDS = {
    "wikisource_ershinianmuduzhiguai_001": 0.50,
    "wikisource_guanchangxianxingji_001": 0.45,
}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_thresholds(raw: str | None) -> dict[str, float]:
    thresholds = dict(DEFAULT_THRESHOLDS)
    if not raw:
        return thresholds

    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Bad threshold item: {item}")
        source_id, value = item.split("=", 1)
        thresholds[source_id.strip()] = float(value.strip())

    return thresholds


def load_mined_hard_negatives(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        rows.extend(read_jsonl(path))
    return rows


def make_hard_negative_training_rows(
    mined_rows: list[dict],
    thresholds: dict[str, float],
    repeat: int,
    default_threshold: float,
) -> list[dict]:
    selected = []

    for row in mined_rows:
        source_id = row.get("source_id")
        split = row.get("_split") or row.get("split")
        strategy = row.get("strategy")
        model_score = float(row.get("model_score", 0.0))
        threshold = thresholds.get(source_id, default_threshold)

        if split != "train":
            continue
        if strategy != "random_distant_negative":
            continue
        if model_score < threshold:
            continue

        base = dict(row)

        # Keep training schema aligned with the existing train rows.
        base["final_score"] = 0.0
        base["label_source"] = "mined_hard_negative_v1"
        base["hard_negative"] = True
        base["hard_negative_source_model_score"] = model_score
        base["hard_negative_threshold"] = threshold

        # Remove analysis-only fields that are not needed for training.
        base.pop("model_score", None)
        base.pop("_split", None)

        for i in range(repeat):
            out = dict(base)
            out["pair_id"] = f"{base.get('pair_id')}#hn{i + 1}"
            selected.append(out)

    return selected


def summarize(rows: list[dict], title: str) -> None:
    by_source = {}
    by_strategy = {}
    for r in rows:
        by_source[r.get("source_id")] = by_source.get(r.get("source_id"), 0) + 1
        by_strategy[r.get("strategy")] = by_strategy.get(r.get("strategy"), 0) + 1

    print()
    print(title)
    print("rows:", len(rows))
    print("sources:")
    for k, v in sorted(by_source.items(), key=lambda x: str(x[0])):
        print(f"  {k}: {v}")
    print("strategies:")
    for k, v in sorted(by_strategy.items(), key=lambda x: str(x[0])):
        print(f"  {k}: {v}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a new training split by adding mined hard negatives to train only."
    )
    parser.add_argument(
        "--base-split-dir",
        default=str(ROOT / "data" / "train_splits" / "rule_only_v1"),
    )
    parser.add_argument(
        "--output-split-dir",
        default=str(ROOT / "data" / "train_splits" / "rule_hn_v1"),
    )
    parser.add_argument(
        "--hard-negative-jsonl",
        action="append",
        required=True,
        help="Path to mined hard negative jsonl. Repeat this argument for multiple files.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Repeat each selected hard negative this many times in train.",
    )
    parser.add_argument(
        "--default-threshold",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--thresholds",
        default=None,
        help="Optional comma-separated source thresholds, e.g. source_a=0.5,source_b=0.45",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_split_dir)
    out_dir = Path(args.output_split_dir)

    train_path = base_dir / "train.jsonl"
    valid_path = base_dir / "valid.jsonl"
    test_path = base_dir / "test.jsonl"

    if not train_path.exists():
        raise FileNotFoundError(train_path)
    if not valid_path.exists():
        raise FileNotFoundError(valid_path)
    if not test_path.exists():
        raise FileNotFoundError(test_path)

    thresholds = parse_thresholds(args.thresholds)
    mined_paths = [Path(p) for p in args.hard_negative_jsonl]
    mined_rows = load_mined_hard_negatives(mined_paths)

    base_train = read_jsonl(train_path)
    hard_train = make_hard_negative_training_rows(
        mined_rows=mined_rows,
        thresholds=thresholds,
        repeat=args.repeat,
        default_threshold=args.default_threshold,
    )

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    new_train = base_train + hard_train
    write_jsonl(out_dir / "train.jsonl", new_train)

    # valid/test are copied unchanged to avoid evaluation contamination.
    shutil.copy2(valid_path, out_dir / "valid.jsonl")
    shutil.copy2(test_path, out_dir / "test.jsonl")

    metadata = {
        "base_split_dir": str(base_dir),
        "output_split_dir": str(out_dir),
        "hard_negative_files": [str(p) for p in mined_paths],
        "thresholds": thresholds,
        "default_threshold": args.default_threshold,
        "repeat": args.repeat,
        "base_train_rows": len(base_train),
        "selected_hard_negative_rows_after_repeat": len(hard_train),
        "selected_unique_hard_negatives": len(hard_train) // args.repeat if args.repeat else 0,
        "new_train_rows": len(new_train),
        "valid_rows": sum(1 for _ in (out_dir / "valid.jsonl").open("r", encoding="utf-8")),
        "test_rows": sum(1 for _ in (out_dir / "test.jsonl").open("r", encoding="utf-8")),
        "notes": [
            "Only train split receives mined hard negatives.",
            "valid/test are copied unchanged from base split.",
            "Hard negatives are selected from mined rows with _split=train only.",
        ],
    }

    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summarize(base_train, "BASE TRAIN")
    summarize(hard_train, "ADDED HARD NEGATIVES AFTER REPEAT")
    summarize(new_train, "NEW TRAIN")
    print()
    print("Wrote:", out_dir)
    print("Metadata:", out_dir / "metadata.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
