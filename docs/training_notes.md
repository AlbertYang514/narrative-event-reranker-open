# Training Notes

## rule_only_v1

Purpose:

Train a minimal Chinese narrative event relevance cross-encoder using rule-only weak supervision.

This is not a generative model and not an LLM.  
The first model is a small regression ranker.

## Input

Model input:

- current_text
- candidate_text

The weak-label strategy is not provided to the model.

## Label source

Rule-only soft labels:

- adjacent_chunk: 0.80
- same_chapter_gap_2 / same_section_gap_2: 0.60
- same_chapter_gap_3 / same_section_gap_3: 0.45
- same_chapter_gap_4 / same_section_gap_4: 0.35
- chapter_boundary / section_boundary: 0.40
- random_distant_negative: 0.00

## Current public training data

- wikisource_wenmingxiaoshi_001
- wikisource_niehaihua_001

Total split rows:

- train: 2525
- valid: 316
- test: 317
- total: 3158

## Important limitation

The current split is rule-heavy and near-neighbor-heavy.

This model should not be interpreted as deep literary understanding.  
It is a first pipeline validation model for learning local narrative continuity signals.

Future data must add:

- hard negatives: nearby but unrelated, same-character but unrelated
- hard positives: distant but causally related, foreshadowing, character-line callbacks

## Private eval policy

Private self-owned / assistant-assisted fiction can only be used for local evaluation or regression.

It must not enter training data or distributed datasets.

## rule_only_v1_cross_encoder_hfl_rbt3_v1

Base model:

- hfl/rbt3

Training setup:

- split: data/train_splits/rule_only_v1
- max_length: 384
- batch_size: 16
- epochs: 3
- lr: 2e-5
- device: one consumer NVIDIA GPU, CUDA (original experiment)
- output: outputs/rule_only_v1_cross_encoder_hfl_rbt3_v1

Final metrics:

Valid:

- mse: 0.08558437005463217
- rmse: 0.29254806452040005
- mae: 0.2305970170543989
- pearson: 0.4530422038730837
- spearman: 0.36087449050821463
- auc_label_ge_0_5: 0.6377038016264072
- pred_mean: 0.346766333457501
- label_mean: 0.5156645626772808

Test:

- mse: 0.13384399665079277
- rmse: 0.365846957963016
- mae: 0.30524002810189504
- pearson: 0.29419965957305005
- spearman: 0.2818037620738358
- auc_label_ge_0_5: 0.6471208434712086
- pred_mean: 0.3086360739334707
- label_mean: 0.5757097856667516

Interpretation:

The first trained baseline shows non-random ranking signal on rule-only labels.

However, predictions are under-calibrated and biased low. The current dataset is near-neighbor-heavy and lacks enough hard negatives / hard positives, so this should be treated as a pipeline baseline rather than a strong narrative understanding model.

## rule_only_v1_cross_encoder_hfl_rbt3_b8_e3

Base model:

- hfl/rbt3

Training setup:

- split: data/train_splits/rule_only_v1
- max_length: 384
- batch_size: 8
- epochs: 3
- lr: 2e-5
- device: one consumer NVIDIA GPU, CUDA (original experiment)

Final metrics:

Valid:

- mse: 0.07772485028420795
- rmse: 0.2787917686808704
- mae: 0.2206317918593229
- pearson: 0.441692696601872
- spearman: 0.3662395941674993
- auc_label_ge_0_5: 0.6437527540760326
- pred_mean: 0.38313448560765656
- label_mean: 0.5156645626772808

Test:

- mse: 0.11306446428583698
- rmse: 0.33625059745052793
- mae: 0.2789057268679295
- pearson: 0.2986714404471493
- spearman: 0.2838133298738802
- auc_label_ge_0_5: 0.6502433090024331
- pred_mean: 0.3514037782675495
- label_mean: 0.5757097856667516

Interpretation:

Compared with the first batch-16 run, this batch-8 run improves MSE/MAE and slightly improves AUC, while Spearman remains roughly similar. The model remains under-calibrated but less severely biased low.

This supports the current conclusion that the pipeline is working and the model is learning rule-only continuity signals, but further gains likely require more and better data rather than longer training on the same split.

## rule_only_v2_cross_encoder_hfl_rbt3_b8_e3

Base model:

- hfl/rbt3

Training setup:

- split: data/train_splits/rule_only_v1
- data version: rule-only v2 expanded public sources
- public sources:
  - wikisource_wenmingxiaoshi_001
  - wikisource_niehaihua_001
  - wikisource_guanchangxianxingji_001
  - wikisource_ershinianmuduzhiguai_001
- total pairs: 7190
- train / valid / test: 5751 / 719 / 720
- max_length: 384
- batch_size: 8
- epochs: 3
- lr: 2e-5
- device: one consumer NVIDIA GPU, CUDA (original experiment)

Final metrics:

Valid:

- mse: 0.08204087755990272
- rmse: 0.286427787688106
- mae: 0.22767133430851408
- pearson: 0.4197686645606034
- spearman: 0.34706576343571954
- auc_label_ge_0_5: 0.6506180871828237
- pred_mean: 0.3771578931228407
- label_mean: 0.524061202049587

Test:

- mse: 0.09407180328851032
- rmse: 0.3067112702339292
- mae: 0.24484323161757654
- pearson: 0.27857622480551136
- spearman: 0.27704601859754907
- auc_label_ge_0_5: 0.6344874650936587
- pred_mean: 0.3864080095016915
- label_mean: 0.5706250065316757

Interpretation:

Compared with the earlier rule-only v1 runs, this expanded-data v2 run improves regression error and score calibration, but does not significantly improve ranking metrics. The main issue is no longer whether training works, but whether the split and negative sampling match the real reranking task.

The v2 split was later found to be poorly balanced: the test set contained only 10 random_distant_negative examples. Therefore this result should be treated as a stable expanded-data baseline, not as a reliable final estimate of reranking quality.

## rule_only_v3_stratified_cross_encoder_hfl_rbt3_b8_e3

Base model:

- hfl/rbt3

Training setup:

- split: data/train_splits/rule_only_v1
- data version: rule-only v3 stratified by source and strategy
- public sources:
  - wikisource_wenmingxiaoshi_001
  - wikisource_niehaihua_001
  - wikisource_guanchangxianxingji_001
  - wikisource_ershinianmuduzhiguai_001
- total pairs: 7190
- train / valid / test: 5746 / 712 / 732
- max_length: 384
- batch_size: 8
- epochs: 3
- lr: 2e-5
- device: one consumer NVIDIA GPU, CUDA (original experiment)

Split notes:

The v3 split is stratified by source_id and strategy. This fixes the v2 issue where the test set contained only 10 random_distant_negative examples.

Final test metrics:

- mse: 0.05955934898447164
- rmse: 0.24404784158945483
- mae: 0.19730022013991502
- pearson: 0.5891474205298752
- spearman: 0.5350667058742814
- auc_label_ge_0_5: 0.7183120283382105
- pred_mean: 0.3980134477383391
- label_mean: 0.4194672179759526

Interpretation:

This is the strongest rule-only baseline so far. Compared with the expanded-data v2 run, v3 substantially improves ranking and regression metrics because the evaluation split is better balanced across sources and sampling strategies.

The result shows that the rbt3 cross-encoder can learn the weakly supervised narrative continuity labels with useful ranking signal. However, the labels are still rule-only and distance-heavy. Future improvements should focus on hard negatives, hard positives, and product-shaped reranking evaluation rather than longer training on the same labels.
