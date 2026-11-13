# Teacher Signal Notes

## Conclusion

A small open-weight reasoning model
(`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`) was tested as a potential teacher
for narrative event relevance labeling. It was not reliable enough to act as
the primary teacher.

It was tested in several modes:

1. JSON generation
2. JSON-prefix generation
3. Multiple-choice generation
4. Logit-based multiple-choice scoring

Generation modes were unstable because the model often produced reasoning text,
malformed JSON, or overly conservative `uncertain` labels. The logit-based
multiple-choice mode was format-stable, but its semantic signal was weak.

On two private held-out works, teacher-only expected scores did not reliably
separate positive narrative pairs from random distant negatives. The observed
AUCs were roughly 0.27 and 0.57. The works and their texts are not distributed.

Therefore, R1 teacher output should be treated as a noisy auxiliary signal only,
not as a ground-truth label source.

## Current training-label policy

The first training baseline uses rule-only weak supervision.

Strategy priors:

- `adjacent_chunk`: 0.80
- `same_chapter_gap_2` / `same_section_gap_2`: 0.60
- `same_chapter_gap_3` / `same_section_gap_3`: 0.45
- `same_chapter_gap_4` / `same_section_gap_4`: 0.35
- `chapter_boundary` / `section_boundary`: 0.40
- `random_distant_negative`: 0.00

Held-out evaluation data remains `training=false` and must not be used for
training.

## Next direction

Use rule-only soft labels to train a minimal baseline ranker. R1 outputs may be
kept for analysis, but should not dominate labels unless later validated against
a human-labeled golden set.
