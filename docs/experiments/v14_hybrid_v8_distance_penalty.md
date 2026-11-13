# v14 Hybrid: v8 + same-source distance penalty

Date: 2026-06-11

## Conclusion

The best current production strategy is not another trained checkpoint.

Use:

base_model = rule_hn_v8_error_replay_from_v7_cross_encoder_hfl_rbt3_b8_e1_lr2e6
scoring_strategy = v14_hybrid_v8_distance_penalty

v14 is a hybrid ranking strategy:

final_score = raw_model_score - distance_penalty

It is not a standalone model checkpoint.

## Base model

Checkpoint used during the original experiment (local, not distributed):

`outputs/rule_hn_v8_error_replay_from_v7_cross_encoder_hfl_rbt3_b8_e1_lr2e6/best`

v8 remains the base model.

## Distance penalty rule

Only apply to same-source / same-book candidates where chunk distance can be parsed.

Rules:

- gap >= 250: penalty = 0.16
- gap >= 140: penalty = 0.12
- gap >= 100: penalty = 0.08
- otherwise: penalty = 0.0

Final score:

hybrid_score = raw_score - distance_penalty(gap)

Recommended output/debug fields:

- raw_score
- distance_gap
- distance_penalty
- hybrid_score
- hybrid_method
- score

score should equal hybrid_score for compatibility with existing sorting.

## Four-book regression

### Guanchang

v8 raw 0.0 max:     0.6826
v14 hybrid 0.0 max: 0.5626
top10 0.0 count:    0
core bad pair:      wikisource_guanchangxianxingji_001:c00006-c00152

### Ershinian

top1 unchanged:     0.6699
v14 hybrid 0.0 max: 0.1998
top20 0.0 count:    0

### Wenming Xiaoshi

top1 label:         0.6
v14 hybrid 0.0 max: 0.3825
top10 0.0 count:    1

### Niehai Hua

top1 label:         0.8
v14 hybrid 0.0 max: 0.4081
top10 0.0 count:    0

## Decision

Training routes v9-v13 did not beat v8 cleanly.

v14 hybrid fixes the main failure mode by using known structural information:

same source + far chunk gap => lower continuity probability

This should be implemented as ranking post-processing before attempting any distillation.
