# v7 Style Contrast Experiment

## Goal

Train a stronger narrative-event reranker for Coastline by reducing false positives caused by style similarity.

The main target is not higher benchmark numbers alone. The goal is to teach the model:

- same style does not mean same narrative chain
- same author does not mean relevant context
- same book but distant event should not be ranked too high
- cross-author modern baihua similarity should not become false relevance

## Base

Base split:

- `data/train_splits/rule_hn_v6_public_modern_baihua_v1`

Base model:

- `rule_hn_v6_public_modern_baihua_from_v5_cross_encoder_hfl_rbt3_b8_e2/best`

## Added data

v7 adds 3000 train-only augmentation rows.

### Qing style contrast: 1000 rows

File:

- `data/augmentation/v7_qing_style_contrast_1000.jsonl`

Strategies:

- `v7_qing_near_positive`: 300
- `v7_qing_same_book_far_negative`: 500
- `v7_qing_cross_book_same_style_negative`: 200

### Modern baihua style contrast: 2000 rows

File:

- `data/augmentation/v7_modern_style_contrast_2000.jsonl`

Strategies:

- `v7_modern_near_positive`: 700
- `v7_modern_same_author_cross_story_negative`: 500
- `v7_modern_cross_author_same_style_negative`: 500
- `v7_modern_far_negative`: 300

Modern sources include:

- Lu Xun: Nahan / Panghuang / Gushi Xinbian
- Xu Dishan: Shangrenfu
- Yu Dafu: Chenlun
- Xiao Hong: Hulanhe Zhuan

Note: Xiao Hong Hulanhe Zhuan dirty tail chapter was removed before chunk/pair generation.

## Final split

Split:

- `data/train_splits/rule_hn_v7_style_contrast_v1`

Counts:

- base train: 7514
- qing augmentation: 1000
- modern augmentation: 2000
- train total: 10514
- valid: 712
- test: 732
- total rows: 11958
- unique pair ids: 11958
- overlap across train/valid/test: 0
- missing current/candidate text: 0

Train label_source:

- `rule_only`: 5746
- `rule_only_public_modern_baihua_v0`: 1515
- `v7_style_contrast`: 3000
- `v5_error_driven`: 148
- `mined_hard_negative_v1`: 105

Train score distribution:

- 0.0: 4835
- 0.8: 1790
- 0.6: 1496
- 0.45: 1228
- 0.35: 955
- 0.4: 210

Mean train score: 0.3139

## Training plan

Recommended first run:

- base model: v6 best
- split: `rule_hn_v7_style_contrast_v1`
- lr: `3e-6`
- epochs: `2`
- batch size: `8`
- max length: `512`
- seed: `42`

Output:

- `outputs/rule_hn_v7_style_contrast_from_v6_cross_encoder_hfl_rbt3_b8_e2_lr3e6`

## Risk

v7 adds many negative examples.

Expected benefit:

- fewer high-score false positives caused by same style / same author / same book distant similarity

Main risk:

- model may become too conservative
- pred_mean may drop
- positive adjacency scores may also be suppressed

If v7 becomes too conservative, rerun the same split with softer training fire, e.g.:

- lr `2e-6`, epochs `1`
- or lr `2e-6`, epochs `2`

Do not immediately discard the data before checking whether the failure is caused by training intensity.

## Evaluation priorities

v7 success should be judged by:

1. old Qing valid/test regression should not collapse
2. modern baihua eval should not collapse
3. pred_mean should not drop too far
4. rank-case false positives should improve
5. high-score label=0 examples in Ershinian / Guanchang should decrease

v7 does not need to dominate every metric to be useful. It should mainly reduce context pollution for Coastline.
