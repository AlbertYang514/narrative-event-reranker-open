# Narrative Event Reranker

A small research codebase for training and evaluating a lightweight Chinese
long-form **narrative continuity / relevance reranker**. Given a current text
passage and a candidate passage, the model predicts a score in roughly `0..1`.

- Near `1`: the candidate is likely a natural narrative continuation / relevant
  context for the current passage.
- Near `0`: the candidate is likely narratively disconnected.

This is a cross-encoder regression model. It is not a chatbot, a text generator,
or a writing model.

> **Code-only release.** This repository contains source code, documentation,
> configuration schemas, and tiny synthetic examples only. It does **not**
> contain training/validation/test corpora, private evaluation data, model
> weights, optimizer states, exported ONNX/Core ML artifacts, or any Coastline
> proprietary release component.

## Project status

The experiments below were run by the author during research. The repository
does not ship a checkpoint, so the numbers cannot be reproduced without
obtaining the base model and rebuilding the data pipeline from lawful sources.

| Component | Value |
|---|---|
| Main experimental checkpoint | v8 (not distributed) |
| Recommended ranking strategy | `v14_hybrid` — raw score + same-source chunk-distance penalty |
| Base architecture | `hfl/chinese-roberta-wwm-ext` fine-tuned as a cross-encoder (`num_labels=1`) |
| Input | `current_text` + `candidate_text` |
| Output | A single regression score |

## What is included

- Chunking, source import and manifest validation scripts
- Rule-based weak-label generation
- Hard-negative mining and split construction
- Cross-encoder training and evaluation
- Candidate ranking with raw or `v14_hybrid` scoring
- Optional local/held-out evaluation tooling
- Public source metadata examples
- Tiny synthetic examples and unit tests

## What is not included

- No training, validation or test JSONL corpus
- No private / held-out evaluation data, text or identifiers
- No model weights (`*.pt`, `*.pth`, `*.bin`, `*.safetensors`, `*.ckpt`,
  `*.onnx`) or optimizer states
- No third-party copyrighted text dumps
- No Coastline source code or bundled artifacts

See [`docs/data_policy.md`](docs/data_policy.md) for the data rules.

## Installation

Python 3.10+ is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch installation depends on your platform and accelerator. If the default
`pip install torch` is not appropriate, install the build recommended by
<https://pytorch.org/> first, then install the remaining requirements.

## Synthetic smoke example

`examples/synthetic/` contains a tiny, fully synthetic split with the same JSONL
schema as a real split. It is intended for interface tests only, not for
measuring model quality.

```bash
python3 -m unittest discover -s tests -v
python3 scripts/inspect_train_splits.py --split-dir examples/synthetic
```

To exercise training code without downloading a checkpoint, pass any local
compatible Hugging Face model directory:

```bash
python3 scripts/train_cross_encoder.py \
  --model-name-or-path /path/to/local/model \
  --split-dir examples/synthetic \
  --output-dir /tmp/reranker-smoke \
  --epochs 1 \
  --batch-size 2 \
  --max-train-steps 2 \
  --max-length 64 \
  --ignore-mismatched-sizes
```

## Data pipeline

The general pipeline is:

1. Record each source in a manifest (`data/manifests/*.jsonl`).
2. Import text with `scripts/import_manifest_sources.py` or
   `scripts/fetch_wikisource_work.py`.
3. Normalize Chinese variants with `scripts/normalize_zh.py`.
4. Build chunks with `scripts/build_chunks.py`.
5. Build candidate pairs with `scripts/build_pair_candidates.py`.
6. Create weak labels with `scripts/build_rule_soft_labels.py`.
7. Build train/valid/test splits with `scripts/build_train_splits.py`.

The repository does not redistribute raw source text. Public-domain source
metadata and import notes are kept for reproducibility; users must fetch the
text themselves under the relevant terms.

## Training

```bash
python3 scripts/train_cross_encoder.py \
  --model-name-or-path hfl/chinese-roberta-wwm-ext \
  --split-dir data/train_splits/rule_only_v1 \
  --output-dir outputs/my_run \
  --max-length 384 \
  --batch-size 8 \
  --epochs 1 \
  --lr 2e-5 \
  --device auto
```

The base model is downloaded by Hugging Face tooling and is not part of this
repository. Check the base model's license before use.

## Evaluation

```bash
python3 scripts/evaluate_cross_encoder.py \
  --model-path outputs/my_run/best \
  --split-dir data/train_splits/rule_only_v1 \
  --device auto \
  --output-json /tmp/eval.json
```

## Candidate ranking

```bash
python3 scripts/rank_candidates.py \
  --model-path outputs/my_run/best \
  --split-dir data/train_splits/rule_only_v1 \
  --source-id YOUR_SOURCE_ID \
  --scoring-strategy v14_hybrid
```

`v14_hybrid` subtracts a small penalty for same-source candidates separated by
a large chunk gap:

```text
gap >= 250 -> 0.16
gap >= 140 -> 0.12
gap >= 100 -> 0.08
otherwise  -> 0.00
```

The penalty is post-processing, not a separate model. See
[`docs/experiments/v14_hybrid_v8_distance_penalty.md`](docs/experiments/v14_hybrid_v8_distance_penalty.md).

## Experimental metrics

The v8 experiment reported on its validation/test splits:

| Split | Spearman | Pearson | AUC (label >= 0.5) |
|---|---|---|---|
| valid | 0.5936 | 0.6395 | 0.7685 |
| test | 0.5569 | 0.6112 | 0.7311 |

These are **weakly supervised, rule-heavy** results. Labels are derived from
chunk adjacency and distance priors, and the evaluation splits are built from
the same pipeline. They indicate that the pipeline learns useful ranking
signals; they are not evidence of general narrative understanding, calibrated
probabilities, or production-level quality on arbitrary Chinese text. See
[`docs/model_card_v8.md`](docs/model_card_v8.md) and
[`docs/training_notes.md`](docs/training_notes.md) for caveats.

## License and third-party components

- Repository code: MIT, see [`LICENSE`](LICENSE).
- Base model: `hfl/chinese-roberta-wwm-ext` (Apache-2.0 at the time of the
  experiments). Verify the current model card before redistribution or
  commercial use.
- Teacher model used in early weak-label experiments:
  `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (MIT, with Qwen2.5-derived terms).
  No teacher weights are distributed here.
- Public-domain source texts are **not** redistributed. Source URLs and license
  status are recorded in manifests; users are responsible for re-fetching and
  complying with the source terms.
- Coastline is a separate product. This repository is not a Coastline release
  and does not include Coastline proprietary code, models or bundled artifacts.
  Coastline components remain under their own terms.

## Limitations

- Weak supervision from chunk distance and adjacency can reward superficial
  proximity rather than true narrative continuity.
- Same-source long-distance false positives are a known failure mode; the
  `v14_hybrid` penalty mitigates but does not solve it.
- The model was studied on late-Qing and early-20th-century Chinese narrative
  text. It may not generalise to modern web fiction, non-narrative text, or
  other languages.
- Absolute scores are not calibrated probabilities.
- No checkpoint is published, so the reported metrics are historical
  experimental results, not a claim about a downloadable model.

## Repository layout

```text
configs/              label and example schemas
docs/                 data policy, training notes, model card, experiments
examples/synthetic/   tiny synthetic split for interface tests
scripts/              data, training, evaluation and ranking code
tests/                dependency-light unit tests
```
