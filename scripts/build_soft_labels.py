#!/usr/bin/env python3
"""Convert teacher-labelled pair candidates into soft-label JSONL.

The script is intentionally input-driven: training data and held-out evaluation
data are separate invocations, so the public repository never needs to know the
name or location of a private evaluation set.

Examples:
    python scripts/build_soft_labels.py \
        --input data/teacher_labeled/teacher_batch_001.labeled.jsonl \
        --output data/soft_labels/train/teacher_batch_001.soft_labels.jsonl \
        --training true --usage train_candidate

    python scripts/build_soft_labels.py \
        --input /path/to/heldout.labeled.jsonl \
        --output /path/to/heldout.soft_labels.jsonl \
        --training false --usage local_eval_only
"""

import argparse
import json
from pathlib import Path

CHOICE_SCORE = {
    "A": 0.0,
    "B": 0.4,
    "C": 0.8,
    "D": 1.0,
    "E": 0.2,
}

STRATEGY_PRIOR = {
    "adjacent_chunk": 0.80,
    "same_chapter_gap_2": 0.60,
    "same_chapter_gap_3": 0.45,
    "same_chapter_gap_4": 0.35,
    "chapter_boundary": 0.40,
    "same_section_gap_2": 0.60,
    "same_section_gap_3": 0.45,
    "same_section_gap_4": 0.35,
    "section_boundary": 0.40,
    "random_distant_negative": 0.00,
}


def expected_score(choice_probs: dict) -> float:
    total = 0.0
    mass = 0.0
    for choice, score in CHOICE_SCORE.items():
        p = float(choice_probs.get(choice, 0.0))
        total += p * score
        mass += p
    if mass <= 0:
        return 0.0
    return total / mass


def text_of(item) -> str:
    if isinstance(item, dict):
        return item.get("text", "")
    return ""


def convert_file(
    input_path: Path,
    output_path: Path,
    training: bool,
    usage: str,
    teacher_weight: float,
) -> int:
    if not 0.0 <= teacher_weight <= 1.0:
        raise SystemExit("--teacher-weight must be between 0 and 1")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for line_no, line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        r = json.loads(line)

        strategy = r.get("strategy")
        prior = STRATEGY_PRIOR.get(strategy)
        if prior is None:
            raise ValueError(f"Unknown strategy {strategy!r} at {input_path}:{line_no}")

        teacher = r.get("teacher_label") or {}
        probs = teacher.get("choice_probs") or {}
        t_score = expected_score(probs)

        final = (1.0 - teacher_weight) * prior + teacher_weight * t_score

        rows.append({
            "pair_id": r.get("pair_id"),
            "source_id": r.get("source_id"),
            "dataset_id": r.get("dataset_id"),
            "strategy": strategy,
            "current_text": text_of(r.get("current")),
            "candidate_text": text_of(r.get("candidate")),
            "rule_score": round(prior, 6),
            "teacher_choice": teacher.get("choice"),
            "teacher_label": teacher.get("label"),
            "teacher_confidence": teacher.get("confidence"),
            "teacher_expected_score": round(t_score, 6),
            "final_score": round(final, 6),
            "training": training,
            "usage": usage,
            "teacher_model": r.get("teacher_model"),
            "prompt_style": r.get("prompt_style"),
        })

    with output_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {len(rows)} rows -> {output_path}")
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="Teacher-labelled JSONL input")
    ap.add_argument("--output", required=True, help="Soft-label JSONL output")
    ap.add_argument("--training", choices=["true", "false"], default="true")
    ap.add_argument("--usage", default=None, help="Usage tag, e.g. train_candidate or local_eval_only")
    ap.add_argument("--teacher-weight", type=float, default=0.15)
    args = ap.parse_args()

    training = args.training == "true"
    usage = args.usage or ("train_candidate" if training else "local_eval_only")
    if not training and usage == "train_candidate":
        raise SystemExit("Held-out data must not use the train_candidate usage tag")

    convert_file(
        Path(args.input),
        Path(args.output),
        training=training,
        usage=usage,
        teacher_weight=args.teacher_weight,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
