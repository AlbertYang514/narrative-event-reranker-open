# Synthetic Example

This directory contains a tiny, fully synthetic narrative split. The sentences
were written for this repository and do not reproduce any published work.

It exists so that:

- the JSONL schema can be inspected without downloading a corpus;
- data-pipeline and scoring tests can run offline;
- `inspect_train_splits.py` has a safe target.

It is **not** a benchmark. Do not train a model on it and report the result as a
quality metric.

```bash
python3 scripts/inspect_train_splits.py --split-dir examples/synthetic --examples 1
```

Files:

- `train.jsonl` — 8 rows
- `valid.jsonl` — 3 rows
- `test.jsonl` — 3 rows
