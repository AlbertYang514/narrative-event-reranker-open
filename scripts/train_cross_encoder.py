from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


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
        # pair_id commonly looks like source:c00006-c00152
        return abs(a1 - a0)
    except ValueError:
        return None


def with_meta_prefix(row: dict[str, Any], field: str) -> str:
    text = str(row.get(field, ""))
    pair_id = str(row.get("pair_id", ""))
    source_id = str(row.get("source_id", ""))
    dist = chunk_distance(pair_id)
    dist_text = "unknown" if dist is None else str(dist)

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
            "strategy": row.get("strategy", row.get("label_strategy", "")),
        }


@dataclass
class Collator:
    tokenizer: Any
    max_length: int

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = self.tokenizer(
            [x["current_text"] for x in batch],
            [x["candidate_text"] for x in batch],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor([x["label"] for x in batch], dtype=torch.float32)
        return encoded


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

    if len(set(np.round(label_arr, 8))) > 1 and len(set(np.round(pred_arr, 8))) > 1:
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
        "mse": float(mse),
        "rmse": float(math.sqrt(mse)),
        "mae": float(mae),
        "pearson": pearson,
        "spearman": spearman,
        "auc_label_ge_0_5": auc,
        "pred_mean": float(pred_arr.mean()),
        "label_mean": float(label_arr.mean()),
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--split-dir", default="data/train_splits/rule_only_v1")
    parser.add_argument("--output-dir", default="outputs/rule_only_v1_cross_encoder")
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--eval-every-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--use-meta-prefix", action="store_true", help="Prefix current/candidate text with lightweight pair metadata")
    parser.add_argument(
        "--ignore-mismatched-sizes",
        action="store_true",
        help="Allow loading a base checkpoint whose classification head differs (useful for tiny smoke models).",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    split_dir = Path(args.split_dir)
    output_dir = Path(args.output_dir)

    train_rows = load_jsonl(split_dir / "train.jsonl")
    valid_rows = load_jsonl(split_dir / "valid.jsonl")
    test_rows = load_jsonl(split_dir / "test.jsonl")

    print(f"train rows: {len(train_rows)}")
    print(f"valid rows: {len(valid_rows)}")
    print(f"test rows: {len(test_rows)}")

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

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=1,
        problem_type="regression",
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
    )
    model.to(device)

    collator = Collator(tokenizer=tokenizer, max_length=args.max_length)

    train_loader = DataLoader(PairDataset(train_rows, use_meta_prefix=args.use_meta_prefix), batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    valid_loader = DataLoader(PairDataset(valid_rows, use_meta_prefix=args.use_meta_prefix), batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(PairDataset(test_rows, use_meta_prefix=args.use_meta_prefix), batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    total_steps = len(train_loader) * args.epochs
    if args.max_train_steps > 0:
        total_steps = min(total_steps, args.max_train_steps)

    warmup_steps = int(total_steps * args.warmup_ratio)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "run_config.json", vars(args) | {
        "resolved_device": str(device),
        "train_rows": len(train_rows),
        "valid_rows": len(valid_rows),
        "test_rows": len(test_rows),
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
    })

    print("initial valid evaluation...")
    valid_init = evaluate(model, valid_loader, device, "valid@init")
    write_json(output_dir / "metrics_valid_init.json", valid_init)
    print(json.dumps({"valid_init": valid_init}, ensure_ascii=False, indent=2))

    global_step = 0
    best_valid_mse = float("inf")

    model.train()
    for epoch in range(args.epochs):
        progress = tqdm(train_loader, desc=f"train epoch {epoch + 1}/{args.epochs}")
        for batch in progress:
            y = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            logits = model(**batch).logits.squeeze(-1)
            loss = F.mse_loss(logits, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            progress.set_postfix({"loss": f"{loss.item():.4f}", "step": global_step})

            if args.eval_every_steps > 0 and global_step % args.eval_every_steps == 0:
                valid_metrics = evaluate(model, valid_loader, device, f"valid@{global_step}")
                write_json(output_dir / f"metrics_valid_step_{global_step}.json", valid_metrics)
                print(json.dumps({f"valid_step_{global_step}": valid_metrics}, ensure_ascii=False, indent=2))

                if valid_metrics["mse"] < best_valid_mse:
                    best_valid_mse = valid_metrics["mse"]
                    best_dir = output_dir / "best"
                    model.save_pretrained(best_dir)
                    tokenizer.save_pretrained(best_dir)
                    write_json(best_dir / "metrics.json", valid_metrics)

                model.train()

            if args.max_train_steps > 0 and global_step >= args.max_train_steps:
                break

        if args.max_train_steps > 0 and global_step >= args.max_train_steps:
            break

    print("final valid evaluation...")
    valid_final = evaluate(model, valid_loader, device, "valid@final")
    write_json(output_dir / "metrics_valid_final.json", valid_final)
    print(json.dumps({"valid_final": valid_final}, ensure_ascii=False, indent=2))

    print("final test evaluation...")
    test_final = evaluate(model, test_loader, device, "test@final")
    write_json(output_dir / "metrics_test_final.json", test_final)
    print(json.dumps({"test_final": test_final}, ensure_ascii=False, indent=2))

    final_dir = output_dir / "final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    print(f"saved final model to: {final_dir}")


if __name__ == "__main__":
    main()
