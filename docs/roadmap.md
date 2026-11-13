# Roadmap

Phase 0:
- Define labels and schemas.
- Build source manifest.
- Create 50-100 private golden samples.

Phase 1:
- Collect clean public-domain/open-license Chinese narrative texts.
- Clean and split chapters.
- Generate candidate event pairs with rules.

Phase 2:
- Run local open-weight 1.5B teacher to label event pairs.
- Validate JSON labels and evidence spans.
- Build train/valid/test datasets.

Phase 3:
- Train a small 20M-100M event relevance ranker.
- Compare against rule-matrix baseline.

Phase 4:
- Export ONNX/Core ML.
- Consider Coastline experimental integration only if evaluation wins.
