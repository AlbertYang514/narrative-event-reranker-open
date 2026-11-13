# Experiments index

Chronological summary of reranker experiments. Each entry links to its detailed notes.

## v3 (baseline)

First cross-encoder fine-tune. Established the training pipeline.  
Not competitive vs later checkpoints; kept for historical reference only.

## v4

Intermediate. Replaced by v5.

## v5 error-driven

Added error-driven hard negatives from v3/v4 ranking errors.  
First checkpoint that meaningfully suppressed known false positives.

## [v6 public modern baihua](v6_public_modern_baihua_eval.md)

- **Added:** Public-domain modern baihua training data (Lu Xun, Xu Dishan, Yu Dafu, Xiao Hong).
- **Conclusion:** Modern eval improved substantially without damaging old Qing-style evaluation.  
- **Status:** Stable fallback.

## [v7 style contrast](v7_style_contrast.md)

- **Added:** 3000 style-contrast augmentation rows (same-author cross-story negatives, cross-author same-style negatives, same-book far-distance negatives).
- **Goal:** Reduce false positives caused by style similarity rather than narrative continuity.
- **Conclusion:** Stronger style discrimination. Valid fallback if v8 proves too aggressive.  
- **Status:** Strong fallback.

## [v8 error replay](v8_error_replay.md)

- **Added:** 372 error-replay augmentation rows over v7 split (positive keepalive + false-positive replay from v7 ranking failures).
- **Conclusion:** Best overall metrics. Ershinian case mostly fixed. Guanchang case partially improved but still a known limitation.  
- **Status:** **Current main checkpoint.**

## v9–v13 (training routes)

Various training strategies attempted after v8. None beat v8 cleanly on the core four-book regression.  
Not recommended for production. Details omitted for brevity.

## [v14 hybrid distance penalty](v14_hybrid_v8_distance_penalty.md)

- **What:** Engineering post-processing strategy on top of v8 raw scores.
  ```
  hybrid_score = raw_score - distance_penalty(gap)
  distance_penalty:  gap >= 250 → 0.16,  gap >= 140 → 0.12,  gap >= 100 → 0.08,  else 0.0
  ```
- **Penalty applies only to same-source / same-book candidate pairs.**
- **Conclusion:** Effectively suppresses same-source long-distance false positives without training a new model.  
- **Status:** **Current recommended ranking strategy.**

## [v14 held-out evaluation](v14_heldout_eval.md)

- Notes on how `v14_hybrid` behaved on two private held-out works.
- The works, texts, paths and pair identifiers are **not** distributed.
- Conclusion: v14 reduces long-distance false positives while preserving
  near-neighbour results. The evidence is qualitative, not a public benchmark.

## Model card

See [`../model_card_v8.md`](../model_card_v8.md) for the v8 model card with metrics, limitations, and Coastline integration notes.
