import json
from pathlib import Path

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

def text_of(item):
    if isinstance(item, dict):
        return item.get("text", "")
    return ""

def convert_pair_file(input_path: Path, output_dir: Path):
    rows_out = []

    for line in input_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        r = json.loads(line)
        strategy = r.get("strategy")
        if strategy not in STRATEGY_PRIOR:
            raise ValueError(f"Unknown strategy {strategy!r} in {input_path}")

        source_id = r.get("source_id")
        if not source_id:
            # pair_id usually starts with source_id:
            source_id = str(r.get("pair_id", input_path.stem)).split(":")[0]

        score = STRATEGY_PRIOR[strategy]

        rows_out.append({
            "pair_id": r.get("pair_id"),
            "source_id": source_id,
            "dataset_id": r.get("dataset_id"),
            "strategy": strategy,
            "current_text": text_of(r.get("current")),
            "candidate_text": text_of(r.get("candidate")),
            "rule_score": score,
            "teacher_choice": None,
            "teacher_label": None,
            "teacher_confidence": None,
            "teacher_expected_score": None,
            "final_score": score,
            "training": True,
            "usage": "train_candidate",
            "label_source": "rule_only",
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{rows_out[0]['source_id']}.soft_labels.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {len(rows_out)} rows -> {out_path}")

def main():
    output_dir = Path("data/soft_labels/train")
    inputs = sorted(Path("data/datasets").glob("*.pair_candidates.jsonl"))

    if not inputs:
        raise SystemExit("No pair candidate files found.")

    for p in inputs:
        print("convert", p)
        convert_pair_file(p, output_dir)

if __name__ == "__main__":
    main()
