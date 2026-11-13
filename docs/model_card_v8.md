# Model card: v8 cross-encoder reranker

## Model identity

- **Name:** `rule_hn_v8_error_replay_from_v7_cross_encoder_hfl_rbt3_b8_e1_lr2e6`
- **Base architecture:** `hfl/chinese-roberta-wwm-ext` fine-tuned as a cross-encoder
- **Task:** Regression — output a single scalar (0–1 narrative continuity / relevance score)
- **Input:** `current_text` + `candidate_text`
- **Output:** Raw logit (higher = more likely narratively continuous)
- **Training code:** `scripts/train_cross_encoder.py`
- **Date:** 2026

## Intended use

- Ranking candidate text chunks by narrative continuity to a given current narrative context.
- Serving as a **Coastline local context gatekeeper**: filtering / re-ranking context candidates before the API writes the formal analysis.
- This model **guards the door**. It does not generate text, does not converse, and does not determine truth.

## Out-of-scope / non-use

- Not a chatbot or conversational model.
- Not a text generator.
- Not a factuality / truthfulness classifier.
- Not a toxicity or safety filter.
- Not intended for scoring creative writing quality.
- Not intended as a standalone decision-maker without human confirmation in Coastline.

## Training data

- **License:** Public-domain Chinese texts from Wikisource, open-license sources, and self-owned texts.
- **Sources:** Late-Qing chaptered fiction, early-20th-century modern baihua fiction and prose (Lu Xun, Xu Dishan, Yu Dafu, Xiao Hong, Zhu Ziqing, Lai He, etc.).
- **No** OpenAI / Claude / commercial API output used as formal training data.
- **No** copyrighted modern web fiction.
- Source metadata is tracked in `data/manifests/sources.public_domain.jsonl`.
- **No source text or training corpus is redistributed in this repository.**

## Metrics (v8 final)

### valid

| Metric | Value |
|--------|-------|
| mse | 0.05512 |
| rmse | 0.23478 |
| mae | 0.18088 |
| pearson | 0.6395 |
| spearman | 0.5936 |
| auc (label ≥ 0.5) | 0.7685 |

### test

| Metric | Value |
|--------|-------|
| mse | 0.05822 |
| rmse | 0.24128 |
| mae | 0.18860 |
| pearson | 0.6112 |
| spearman | 0.5569 |
| auc (label ≥ 0.5) | 0.7311 |

These metrics indicate **engineering-useful** ranking quality for candidate reordering. The model does not achieve perfect calibration, but ranking metrics (pearson, spearman, AUC) are sufficient for the gatekeeper role.

## Known limitations

### Same-source long-distance false positives

When current and candidate come from the same book but are far apart (e.g., chunk gap > 100), the raw v8 score can still be high even when the events are not narratively continuous. This is the main failure mode.

**Mitigation:** v14_hybrid scoring strategy applies a chunk-distance penalty to same-source pairs. See [`docs/experiments/v14_hybrid_v8_distance_penalty.md`](experiments/v14_hybrid_v8_distance_penalty.md).

### Guanchang Xianxing Ji (官场现形记)

Late-Qing chaptered fiction where same-book, same-officialdom-vocabulary chunks score high despite belonging to different narrative chains. v8 improves over v7 but the problem is not fully resolved.

### Over-calibration on modern baihua

Predicted scores on modern baihua splits tend to be higher than labels. Ranking quality is still good, but absolute score thresholds should be used with caution on modern text.

### Narrow domain

The model is trained exclusively on Chinese narrative text (late-Qing to early-20th-century fiction and prose). It may not generalize to:
- Contemporary web fiction
- Non-narrative text (news, technical docs, dialogue transcripts)
- Non-Chinese text

## How to evaluate

```bash
python scripts/evaluate_cross_encoder.py \
  --model-path outputs/rule_hn_v8_error_replay_from_v7_cross_encoder_hfl_rbt3_b8_e1_lr2e6/best \
  --split-dir data/train_splits/rule_hn_v7_style_contrast_v1 \
  --device auto \
  --output-json v8_eval.json
```

Note: Use v7 split (not v8 split) for evaluation, because v8 split contains error-replay examples that should not appear in evaluation.

## Coastline integration plan

This section describes the original research context. It does not grant any
right to Coastline source code or bundled artifacts, and it does not make this
repository part of a Coastline release.

1. Package the v8 model for local inference (ONNX or PyTorch).
2. Integrate as an optional local reranker / candidate reranking layer.
3. Apply v14_hybrid distance penalty as post-processing.
4. The reranker scores candidates; the API writes the formal analysis; the user confirms.
5. The model does not replace the API, does not replace human judgment, and does not make final decisions.

## Distribution

This model card documents an experiment. No checkpoint is distributed with this
repository. Reported metrics are historical experimental results under
weakly-supervised labels, not a guarantee of model capability.
