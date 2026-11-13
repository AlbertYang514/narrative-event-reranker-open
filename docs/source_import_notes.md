# Source Import Notes

## Imported successfully

- `wikisource_wenmingxiaoshi_001`
  - Source: Chinese Wikisource
  - Status: imported, normalized, chunked, pair candidates generated
- `wikisource_niehaihua_001`
  - Source: Chinese Wikisource
  - Status: imported, normalized, chunked, pair candidates generated

## Index-only imports requiring repair

The following sources were fetched as index/table-of-contents pages only. They must not be normalized, chunked, labeled, or used for training until the chapter-link extraction is fixed.

- `wikisource_laocanyouji_001`
- `wikisource_guanchangxianxingji_001`
- `wikisource_ershinianmuduzhiguai_001`

Observed raw sizes were only a few KB, indicating table-of-contents content rather than full novel text.

## Policy

Only sources with full raw text and chapter JSONL content should enter:

- `data/normalized/`
- `data/datasets/`
- `data/soft_labels/train/`
- `data/train_splits/`

Index-only imports are allowed to remain in `data/raw/` and `data/processed/` for debugging, but should be excluded from training.
