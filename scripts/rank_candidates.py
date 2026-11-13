#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

_CHUNK_RE = re.compile(r"[:\-]c(\d+)")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def short_text(text: str, n: int = 160) -> str:
    text = " ".join(str(text).split())
    if len(text) <= n:
        return text
    return text[:n] + "..."


def with_meta_prefix(text: str, role: str, pair: dict | None = None, candidate: dict | None = None) -> str:
    """Add lightweight role/source metadata used by --use-meta-prefix models."""
    source_id = ""
    if isinstance(pair, dict):
        source_id = str(pair.get("source_id") or pair.get("dataset_id") or "")
    if not source_id and isinstance(candidate, dict):
        source_id = str(candidate.get("source_id") or candidate.get("dataset_id") or "")
    return f"[META role={role} source={source_id}]\n{text}"


def parse_chunk_numbers(pair_id: str) -> list[int]:
    """Extract all chunk numbers from a pair_id like 'source:c00595-c00596'."""
    if not pair_id:
        return []
    matches = _CHUNK_RE.findall(str(pair_id))
    try:
        return [int(m) for m in matches]
    except (ValueError, TypeError):
        return []


def compute_chunk_gap(
    current_row: dict,
    candidate_row: dict,
) -> int | None:
    """Compute chunk-index gap between anchor and candidate.

    Priority order (first available wins):

    1. ``metadata.chunk_distance`` on the candidate row.
    2. ``global_chunk_index`` on both rows.
    3. Regex parse of ``pair_id`` fields (fallback).

    Returns ``None`` when all methods fail to produce a number.
    """
    # 1. metadata.chunk_distance (pair-candidate rows)
    metadata = candidate_row.get("metadata")
    if isinstance(metadata, dict):
        cd = metadata.get("chunk_distance")
        if cd is not None:
            try:
                return abs(int(cd))
            except (ValueError, TypeError):
                pass

    # 2. global_chunk_index
    cur_idx = current_row.get("global_chunk_index")
    cand_idx = candidate_row.get("global_chunk_index")
    if cur_idx is not None and cand_idx is not None:
        try:
            return abs(int(cur_idx) - int(cand_idx))
        except (ValueError, TypeError):
            pass

    # 3. pair_id regex fallback
    cur_pid = str(current_row.get("pair_id", ""))
    cand_pid = str(candidate_row.get("pair_id", ""))
    a_chunks = parse_chunk_numbers(cur_pid)
    c_chunks = parse_chunk_numbers(cand_pid)
    if a_chunks and c_chunks:
        return min(abs(a - c) for a in a_chunks for c in c_chunks)

    return None


def compute_distance_penalty(gap: int) -> float:
    """v14 distance-penalty thresholds (checked highest-first)."""
    if gap >= 250:
        return 0.16
    if gap >= 140:
        return 0.12
    if gap >= 100:
        return 0.08
    return 0.0


def apply_scoring_strategy(
    scored: list[dict],
    strategy: str,
    current_pair: dict,
) -> list[dict]:
    """Apply post-inference scoring strategy **in-place**.

    ``raw``
        No-op – keep the raw model logit as ``score``.
    ``v14_hybrid``
        Subtract a same-source distance penalty from the raw score and
        re-sort by the resulting hybrid score.
    """
    if strategy == "raw":
        return scored

    if strategy == "v14_hybrid":
        # Resolve a stable source / book identifier from whichever field is
        # present: source_id first, then dataset_id (local eval rows).
        anchor_id = str(
            current_pair.get("source_id") or current_pair.get("dataset_id") or ""
        )

        for row in scored:
            raw_score = float(row["score"])
            candidate_id = str(
                row.get("source_id") or row.get("dataset_id") or ""
            )

            # Only penalise same-source / same-book pairs.
            if anchor_id and candidate_id and anchor_id == candidate_id:
                gap = compute_chunk_gap(current_pair, row)
            else:
                gap = None

            if gap is not None:
                distance_penalty = compute_distance_penalty(gap)
            else:
                distance_penalty = 0.0

            hybrid_score = raw_score - distance_penalty

            row["raw_score"] = raw_score
            row["distance_gap"] = gap
            row["distance_penalty"] = distance_penalty
            row["hybrid_score"] = hybrid_score
            row["hybrid_method"] = "v14_hybrid_distance_penalty"
            row["score"] = hybrid_score

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored

    raise ValueError(f"Unknown scoring strategy: {strategy}")


def load_split_rows(split_dir: Path) -> list[dict]:
    rows = []
    for name in ("train.jsonl", "valid.jsonl", "test.jsonl"):
        path = split_dir / name
        if path.exists():
            part = read_jsonl(path)
            for r in part:
                r["_split"] = name.replace(".jsonl", "")
            rows.extend(part)
    if not rows:
        raise FileNotFoundError(f"No split rows found in {split_dir}")
    return rows


def select_current_text(rows: list[dict], source_id: str, seed: int, pair_id: str | None) -> tuple[str, dict]:
    pool = [r for r in rows if r.get("source_id") == source_id]
    if not pool:
        raise ValueError(f"No rows for source_id={source_id}")

    if pair_id:
        for r in pool:
            if r.get("pair_id") == pair_id:
                return str(r.get("current_text", "")), r
        raise ValueError(f"pair_id not found for source_id={source_id}: {pair_id}")

    rng = random.Random(seed)

    # Prefer a positive-ish row so current_text is likely a normal narrative chunk.
    preferred = [r for r in pool if float(r.get("final_score", 0.0)) >= 0.6]
    if not preferred:
        preferred = pool

    r = rng.choice(preferred)
    return str(r.get("current_text", "")), r


def build_candidates_from_split(
    rows: list[dict],
    source_id: str,
    current_text: str,
    current_pair: dict,
    max_candidates: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    pool = [r for r in rows if r.get("source_id") == source_id]

    candidates = []
    seen_text = set()

    def add_from_row(r: dict, text_key: str, reason: str):
        text = str(r.get(text_key, ""))
        if not text.strip():
            return
        if text == current_text:
            return
        key = " ".join(text.split())
        if not key or key in seen_text:
            return
        seen_text.add(key)
        candidates.append(
            {
                "text": text,
                "source_id": r.get("source_id"),
                "pair_id": r.get("pair_id"),
                "strategy": r.get("strategy"),
                "label": r.get("final_score"),
                "split": r.get("_split"),
                "reason": reason,
            }
        )

    # 1. Add the known candidate paired with the selected current row.
    add_from_row(current_pair, "candidate_text", "paired_candidate_for_current")

    # 2. Add candidates from rows with the exact same current_text.
    same_current = [r for r in pool if r.get("current_text") == current_text]
    for r in same_current:
        add_from_row(r, "candidate_text", "same_current_candidate")

    # 3. Add high / medium / low label candidates from same source.
    buckets = {
        "label_high": [r for r in pool if float(r.get("final_score", 0.0)) >= 0.6],
        "label_mid": [r for r in pool if 0.35 <= float(r.get("final_score", 0.0)) < 0.6],
        "label_low": [r for r in pool if float(r.get("final_score", 0.0)) <= 0.05],
    }

    for reason, bucket in buckets.items():
        rng.shuffle(bucket)
        for r in bucket[: max(10, max_candidates // 3)]:
            add_from_row(r, "candidate_text", reason)

    # 4. Add random candidates from both current_text and candidate_text fields.
    shuffled = list(pool)
    rng.shuffle(shuffled)
    for r in shuffled:
        add_from_row(r, "candidate_text", "random_candidate_text")
        if len(candidates) >= max_candidates:
            break
        add_from_row(r, "current_text", "random_current_text")
        if len(candidates) >= max_candidates:
            break

    return candidates[:max_candidates]


def score_pairs(
    model,
    tokenizer,
    current_text: str,
    candidates: list[dict],
    device,
    max_length: int,
    batch_size: int,
    use_meta_prefix: bool = False,
    current_pair: dict | None = None,
) -> list[dict]:
    import torch

    scored = []

    model.eval()
    with torch.no_grad():
        for start in range(0, len(candidates), batch_size):
            batch = candidates[start:start + batch_size]
            if use_meta_prefix:
                texts_a = [with_meta_prefix(current_text, "current", pair=current_pair, candidate=c) for c in batch]
                texts_b = [with_meta_prefix(str(c.get("text", "")), "candidate", pair=current_pair, candidate=c) for c in batch]
            else:
                texts_a = [current_text] * len(batch)
                texts_b = [str(c.get("text", "")) for c in batch]

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

            for row, score in zip(batch, logits):
                out = dict(row)
                out["score"] = float(score)
                scored.append(out)

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rank candidate texts from split files using a trained cross-encoder reranker."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--split-dir", default=str(ROOT / "data" / "train_splits" / "rule_only_v1"))
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--pair-id", default=None, help="Optional exact pair_id to use as current_text")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-candidates", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument(
        "--scoring-strategy",
        default="raw",
        choices=["raw", "v14_hybrid"],
        help="Post-inference scoring strategy: raw (default) or v14_hybrid.",
    )
    parser.add_argument(
        "--use-meta-prefix",
        action="store_true",
        help="Prefix current/candidate text with lightweight source metadata (must match training).",
    )
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    split_dir = Path(args.split_dir)
    rows = load_split_rows(split_dir)

    current_text, current_pair = select_current_text(
        rows=rows,
        source_id=args.source_id,
        seed=args.seed,
        pair_id=args.pair_id,
    )

    candidates = build_candidates_from_split(
        rows=rows,
        source_id=args.source_id,
        current_text=current_text,
        current_pair=current_pair,
        max_candidates=args.max_candidates,
        seed=args.seed,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    model.to(device)

    scored = score_pairs(
        model=model,
        tokenizer=tokenizer,
        current_text=current_text,
        candidates=candidates,
        device=device,
        max_length=args.max_length,
        batch_size=args.batch_size,
        use_meta_prefix=args.use_meta_prefix,
        current_pair=current_pair,
    )

    scored = apply_scoring_strategy(
        scored,
        strategy=args.scoring_strategy,
        current_pair=current_pair,
    )

    print("=" * 92)
    print("MODEL:", args.model_path)
    print("DEVICE:", device)
    print("SOURCE:", args.source_id)
    print("SPLIT_DIR:", split_dir)
    print("SCORING:", args.scoring_strategy)
    print("CURRENT PAIR:")
    print("  pair_id:", current_pair.get("pair_id"))
    print("  split:", current_pair.get("_split"))
    print("  strategy:", current_pair.get("strategy"))
    print("  label:", current_pair.get("final_score"))
    print("CURRENT TEXT:")
    print(" ", short_text(current_text, 420))
    print("=" * 92)
    print(f"RANKED CANDIDATES: total={len(scored)} top_k={args.top_k}")
    print()

    for rank, row in enumerate(scored[:args.top_k], start=1):
        if args.scoring_strategy == "v14_hybrid":
            print(
                f"[{rank:02d}] score={row['score']:.4f} "
                f"raw={row.get('raw_score', row['score']):.4f} "
                f"gap={row.get('distance_gap')} "
                f"penalty={row.get('distance_penalty', 0.0):.4f} "
                f"label={row.get('label')} "
                f"strategy={row.get('strategy')} "
                f"split={row.get('split')} "
                f"reason={row.get('reason')}"
            )
        else:
            print(
                f"[{rank:02d}] score={row['score']:.4f} "
                f"label={row.get('label')} "
                f"strategy={row.get('strategy')} "
                f"split={row.get('split')} "
                f"reason={row.get('reason')}"
            )
        print(f"     pair_id={row.get('pair_id')}")
        print("     text:", short_text(row.get("text", ""), 320))
        print()

    if args.output_jsonl:
        out_path = Path(args.output_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for rank, row in enumerate(scored, start=1):
                out = dict(row)
                out["rank"] = rank
                out["current_text"] = current_text
                out["current_pair_id"] = current_pair.get("pair_id")
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
        print(f"Wrote full ranking: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
