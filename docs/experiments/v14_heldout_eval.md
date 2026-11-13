# v14 held-out evaluation notes

Date: 2026-06-11

## Summary

The main experimental checkpoint remains v8. v14 is not a new checkpoint; it is
a ranking strategy:

```text
v14_hybrid = raw_score - same-source distance penalty
```

The held-out material used during the original experiments was private,
self-owned / assistant-assisted fiction. It is **not** distributed with this
repository, and no work title, file path, pair id or per-work metric is included
here. The notes below record only the general shape of the findings so that the
ranking strategy can be understood and re-tested on a user's own held-out set.

## Distance penalty

Only applies to same-source / same-dataset candidates with a parseable chunk gap.

Rules:

- gap >= 250: 0.16
- gap >= 140: 0.12
- gap >= 100: 0.08
- otherwise: 0.0

Distance gap priority:

1. `metadata.chunk_distance`
2. `current.global_chunk_index` vs `candidate.global_chunk_index`
3. `pair_id` fallback parsing

## Held-out observations

Two held-out works were evaluated locally, with 870 and 765 pair rows
respectively. Both contain near-neighbour positives and random distant
negatives. No text or titles are reproduced here.

- The v8 raw score transferred to both held-out works: adjacent and short-gap
  candidates ranked near the top.
- v14 did not change the top near-neighbour candidates, because their distance
  penalty is zero.
- On one held-out work, a long-distance false positive received a high raw
  score and ranked near the top. The distance penalty lowered it by 0.12 while
  preserving the near-neighbour ranking.
- On the other held-out work, only small penalties applied to a minority of
  candidates and the top-20 ordering was unchanged.

These observations are qualitative and based on two non-public works. They are
not a benchmark, not a claim of generalisation, and not reproducible from this
repository alone.

## Decision

- Keep v8 as the main experimental checkpoint.
- Do not train a new checkpoint from the held-out material.
- Do not distill from it.
- Never add held-out text to training data.
- Land v14 as an optional scoring strategy.
- Keep `raw` as the default.
- Use `v14_hybrid` when long-distance same-source false positives are a concern.

## Do not commit

- Local/held-out corpora or source text
- Model checkpoints or optimizer states
- Ranked held-out outputs
- Any identifier that maps an experiment back to a private work
