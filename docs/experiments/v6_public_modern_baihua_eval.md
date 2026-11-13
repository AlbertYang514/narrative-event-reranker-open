# v6 public modern baihua experiment

## Model

`rule_hn_v6_public_modern_baihua_from_v5_cross_encoder_hfl_rbt3_b8_e2`

Base model:

`rule_hn_v5_error_driven_from_v3_cross_encoder_hfl_rbt3_b8_e3/best`

Training split:

`data/train_splits/rule_hn_v6_public_modern_baihua_v1`

Training recipe:

- Base train: `rule_hn_v5_error_driven_v1/train.jsonl`
- Added train-only modern public-domain data: `public_modern_baihua_v0/train.jsonl`
- Main valid/test: unchanged from `rule_hn_v5_error_driven_v1`
- Modern eval: `public_modern_baihua_v0/valid.jsonl` and `test.jsonl`
- Epochs: 2
- LR: `5e-6`
- Batch size: 8
- Max length: 512
- Seed: 42

## Main regression eval: old Qing / rule_hn_v5 test

### v5 test

- mse: 0.0597375
- rmse: 0.2444126
- mae: 0.1924847
- pearson: 0.6013653
- spearman: 0.5427713
- auc_label_ge_0_5: 0.7239643
- pred_mean: 0.4086894

### v6 test

- mse: 0.0580786
- rmse: 0.2409950
- mae: 0.1909624
- pearson: 0.6059310
- spearman: 0.5460486
- auc_label_ge_0_5: 0.7267519
- pred_mean: 0.4059162

### Regression conclusion

Old-task regression passed.

v6 is slightly better than v5 on the unchanged old test. The main value is not the small metric gain, but that adding modern public-domain train rows did not damage the old Qing-style evaluation.

## Modern public-domain eval: public_modern_baihua_v0

### valid

- mse: v5 0.243607 -> v6 0.192380
- rmse: v5 0.493566 -> v6 0.438612
- mae: v5 0.448654 -> v6 0.401431
- pearson: v5 0.661134 -> v6 0.795751
- spearman: v5 0.649131 -> v6 0.753547
- auc_label_ge_0_5: v5 0.894857 -> v6 0.930476
- pred_mean: v5 0.600520 -> v6 0.545711
- label_mean: 0.182162

### test

- mse: v5 0.227671 -> v6 0.179383
- rmse: v5 0.477149 -> v6 0.423537
- mae: v5 0.428146 -> v6 0.382134
- pearson: v5 0.615903 -> v6 0.747088
- spearman: v5 0.628657 -> v6 0.739088
- auc_label_ge_0_5: v5 0.839583 -> v6 0.889881
- pred_mean: v5 0.602353 -> v6 0.548678
- label_mean: 0.208663

### Modern eval conclusion

v6 clearly improves modern public-domain evaluation.

Scores remain over-calibrated on the modern split, but ranking metrics improve substantially. For Coastline reranking, ranking quality is currently more important than absolute score calibration.

## Current model ranking

- Primary candidate: v6
- Fallback: v5
- Old baseline: v3
- Record only: v4, v5.1

## Notes

v6 should replace v5 as the current primary candidate, pending case-rank review.

Next checks:

- Old hard-negative case rank
- Positive anchor suppression
- Modern eval case inspection
- Potential future score calibration if absolute thresholds are needed
