# v8 error replay

v8 = 从 v7 继续训练的错误回放版。

结论：
- v8 是当前主候选。
- v7 是强 fallback。
- v6 是稳定 fallback。
- v8 总体指标小赢。
- 二十年 case 修好。
- 官场 case 仍残留。

路径：
- split: data/train_splits/rule_hn_v8_error_replay_v1
- augmentation: data/augmentation/v8_error_replay_v1.jsonl
- model: outputs/rule_hn_v8_error_replay_from_v7_cross_encoder_hfl_rbt3_b8_e1_lr2e6/best (local, not distributed)

v8 augmentation：
- rows: 372
- unique: 372
- missing: 0
- positive_keepalive: 248
- false_positive_replay: 120
- chrome_negative: 4
- labels: 0.8=133, 0.6=115, 0.0=124

v8 split：
- train total: 12330
- unique pair_ids: 12330
- valid/test 沿用 v7

v7 baseline：
- valid mse/rmse/mae: 0.0553567 / 0.2352800 / 0.1812290
- valid pearson/spearman/auc: 0.6415614 / 0.5956742 / 0.7690927
- test mse/rmse/mae: 0.0588747 / 0.2426411 / 0.1897655
- test pearson/spearman/auc: 0.6126388 / 0.5581617 / 0.7323579

v8 final：
- valid mse/rmse/mae: 0.0546258 / 0.2337216 / 0.1804515
- valid pearson/spearman/auc: 0.6426243 / 0.5951100 / 0.7693607
- test mse/rmse/mae: 0.0579471 / 0.2407220 / 0.1888307
- test pearson/spearman/auc: 0.6138083 / 0.5591578 / 0.7323810

case-rank 注意：
不要用 v8 split 做候选池，因为 v8 split 里有 replay examples。
正确做法：v8 model + v7 split。

二十年：
- v7 里 0.0 false positive 可到约 0.66。
- v8 后 0.0 max score 降到 0.2535。
- top1 为 adjacent，二十年基本修好。

官场：
- v7 同书远距离负例约 0.7182。
- v8 降到 0.6826，但仍排第 6。
- 问题未解决：清末章回体同书、同官场话术、同氛围，但事件链不连续。

v9 建议：
实验名：rule_hn_v9_late_qing_same_book_far_v1

v9 不加现代新资料，先专杀官场型假相关：
- late-Qing chaptered fiction same-book far negatives
- opening-chapter false friends
- same-book same-topic but non-continuous negatives
- same officialdom vocabulary but different event-chain negatives
- adjacent / gap_2 keepalive for chaptered fiction

一句话：
v9 专杀《官场》这种“看起来都在骂官场所以全都像相关”的假相关。
