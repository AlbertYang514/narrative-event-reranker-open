from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"Invalid JSON in {path}:{line_no}: {e}") from e
    return rows


def get_score(row: dict[str, Any]) -> float:
    for key in ("final_score", "score", "label", "teacher_expected_score"):
        if key in row:
            return float(row[key])
    raise KeyError(f"No score field found. Available keys: {sorted(row.keys())}")


_CHUNK_RE = re.compile(r":c(\d+)-c(\d+)")


def chunk_distance(pair_id: str) -> int | None:
    matches = _CHUNK_RE.findall(pair_id or "")
    if len(matches) < 1:
        return None
    try:
        a0, a1 = map(int, matches[0])
        if len(matches) >= 2:
            b0, b1 = map(int, matches[1])
            return abs(b0 - a0)
        return abs(a1 - a0)
    except ValueError:
        return None


def with_meta_prefix(row: dict[str, Any], field: str) -> str:
    text = str(row.get(field, ""))
    pair_id = str(row.get("pair_id", ""))
    source_id = str(row.get("source_id", ""))
    role = "current" if field == "current_text" else "candidate"
    meta = f"[META role={role} source={source_id}]\n"
    return meta + text


class PairDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], use_meta_prefix: bool = False) -> None:
        self.rows = rows
        self.use_meta_prefix = use_meta_prefix

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        current_text = with_meta_prefix(row, "current_text") if self.use_meta_prefix else row["current_text"]
        candidate_text = with_meta_prefix(row, "candidate_text") if self.use_meta_prefix else row["candidate_text"]
        return {
            "current_text": current_text,
            "candidate_text": candidate_text,
            "label": get_score(row),
            "pair_id": row.get("pair_id", ""),
            "source_id": row.get("source_id", ""),
        }


class Collator:
    def __init__(self, tokenizer: Any, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        current = [r["current_text"] for r in batch]
        candidate = [r["candidate_text"] for r in batch]
        labels = torch.tensor([r["label"] for r in batch], dtype=torch.float32)

        encoded = self.tokenizer(
            current,
            candidate,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded["labels"] = labels
        return encoded


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, desc: str) -> dict[str, float]:
    model.eval()
    preds: list[float] = []
    labels: list[float] = []

    for batch in tqdm(loader, desc=desc, leave=False):
        y = batch.pop("labels").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}

        logits = model(**batch).logits.squeeze(-1)
        preds.extend(logits.detach().cpu().float().tolist())
        labels.extend(y.detach().cpu().float().tolist())

    pred_arr = np.array(preds, dtype=np.float64)
    label_arr = np.array(labels, dtype=np.float64)

    mse = mean_squared_error(label_arr, pred_arr)
    mae = mean_absolute_error(label_arr, pred_arr)

    if len(set(np.round(label_arr, 8).tolist())) > 1 and len(set(np.round(pred_arr, 8).tolist())) > 1:
        pearson = float(pearsonr(label_arr, pred_arr).statistic)
        spearman = float(spearmanr(label_arr, pred_arr).statistic)
    else:
        pearson = float("nan")
        spearman = float("nan")

    binary = (label_arr >= 0.5).astype(np.int64)
    if len(set(binary.tolist())) == 2:
        auc = float(roc_auc_score(binary, pred_arr))
    else:
        auc = float("nan")

    return {
        "rows": int(len(label_arr)),
        "mse": float(mse),
        "rmse": float(math.sqrt(mse)),
        "mae": float(mae),
        "pearson": pearson,
        "spearman": spearman,
        "auc_label_ge_0_5": auc,
        "pred_mean": float(pred_arr.mean()),
        "label_mean": float(label_arr.mean()),
        "pred_min": float(pred_arr.min()),
        "pred_max": float(pred_arr.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained cross-encoder reranker against split JSONL files."
    )
    parser.add_argument("--model-path", required=True, help="Path to saved cross-encoder checkpoint directory")
    parser.add_argument("--split-dir", required=True, help="Directory containing train/valid/test JSONL split files")
    parser.add_argument("--output-json", default=None, help="Optional path to write full evaluation results as JSON")
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-meta-prefix", action="store_true", help="Prefix current/candidate text with lightweight pair metadata")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"device: {device}")

    split_dir = Path(args.split_dir)
    valid_rows = load_jsonl(split_dir / "valid.jsonl")
    test_rows = load_jsonl(split_dir / "test.jsonl")

    print(f"valid rows: {len(valid_rows)}")
    print(f"test rows: {len(test_rows)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=1,
        problem_type="regression",
    )
    model.to(device)

    collator = Collator(tokenizer=tokenizer, max_length=args.max_length)

    valid_loader = DataLoader(
        PairDataset(valid_rows, use_meta_prefix=args.use_meta_prefix),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    test_loader = DataLoader(
        PairDataset(test_rows, use_meta_prefix=args.use_meta_prefix),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    result: dict[str, Any] = {
        "model_path": args.model_path,
        "split_dir": str(split_dir),
        "device": str(device),
        "use_meta_prefix": args.use_meta_prefix,
        "valid": evaluate(model, valid_loader, device, "valid"),
        "test": evaluate(model, test_loader, device, "test"),
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
