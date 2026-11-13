#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def short_text(text: str, n: int = 180) -> str:
    text = " ".join(str(text).split())
    if len(text) <= n:
        return text
    return text[:n] + "..."


def load_split_rows(split_dir: Path, splits: list[str]) -> list[dict]:
    rows = []
    for split in splits:
        path = split_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        for row in read_jsonl(path):
            row["_split"] = split
            rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No rows found in {split_dir} for splits={splits}")
    return rows


def select_candidates(
    rows: list[dict],
    source_id: str | None,
    strategy: str,
    max_label: float,
) -> list[dict]:
    out = []
    for r in rows:
        if source_id and r.get("source_id") != source_id:
            continue
        if r.get("strategy") != strategy:
            continue
        score = float(r.get("final_score", 0.0))
        if score <= max_label:
            out.append(r)
    return out


def score_rows(
    model,
    tokenizer,
    rows: list[dict],
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> list[dict]:
    scored = []

    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            texts_a = [str(r.get("current_text", "")) for r in batch]
            texts_b = [str(r.get("candidate_text", "")) for r in batch]

            encoded = tokenizer(
                texts_a,
                texts_b,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}

            outputs = model(**encoded)
            logits = outputs.logits.squeeze(-1).detach().cpu().tolist()
            if isinstance(logits, float):
                logits = [logits]

            for row, pred in zip(batch, logits):
                out = dict(row)
                out["model_score"] = float(pred)
                scored.append(out)

    scored.sort(key=lambda x: x["model_score"], reverse=True)
    return scored


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Mine high-scoring random negatives for hard-negative review.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--split-dir", default=str(ROOT / "data" / "train_splits" / "rule_only_v1"))
    parser.add_argument("--source-id", default=None)
    parser.add_argument("--splits", default="train,valid,test", help="Comma-separated splits")
    parser.add_argument("--strategy", default="random_distant_negative")
    parser.add_argument("--max-label", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-jsonl", required=True)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    rows = load_split_rows(Path(args.split_dir), splits=splits)
    candidates = select_candidates(
        rows=rows,
        source_id=args.source_id,
        strategy=args.strategy,
        max_label=args.max_label,
    )

    print("=" * 96)
    print("MODEL:", args.model_path)
    print("DEVICE:", device)
    print("SPLIT_DIR:", args.split_dir)
    print("SOURCE:", args.source_id or "<all>")
    print("SPLITS:", splits)
    print("STRATEGY:", args.strategy)
    print("CANDIDATES:", len(candidates))
    print("=" * 96)

    if not candidates:
        raise SystemExit("No candidates found.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    model.to(device)

    scored = score_rows(
        model=model,
        tokenizer=tokenizer,
        rows=candidates,
        device=device,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    top = scored[: args.top_k]
    out_path = Path(args.output_jsonl)
    write_jsonl(out_path, top)

    print(f"Wrote top {len(top)} -> {out_path}")
    print()
    print("TOP HARD NEGATIVE CANDIDATES")
    print()

    for i, r in enumerate(top[:20], start=1):
        print(
            f"[{i:02d}] model_score={r['model_score']:.4f} "
            f"label={r.get('final_score')} "
            f"split={r.get('_split')} "
            f"source={r.get('source_id')} "
            f"pair_id={r.get('pair_id')}"
        )
        print("CURRENT:  ", short_text(r.get("current_text", ""), 260))
        print("CANDIDATE:", short_text(r.get("candidate_text", ""), 260))
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
