# Data Policy

This is a **code-only** repository. It does not distribute raw text, processed
corpora, training/validation/test splits, private evaluation data, or model
weights.

Allowed formal training sources:
- Public-domain texts with recorded source metadata.
- Open-license texts whose license permits training / derivative use.
- Fully self-authored texts.
- Synthetic data generated locally or from allowed open-weight teacher models.
- Labels produced by open-weight models whose license permits derivative use.

Not allowed for formal training:
- Modern copyrighted web fiction.
- Pirated text dumps.
- Unknown-source corpora.
- OpenAI / Claude / other commercial API outputs as formal training data.
- Assistant-assisted fiction as public/distributed training data.

Held-out evaluation only:
- Co-authored / assistant-assisted fiction may be used as local evaluation or
  regression tests, but must not be included in formal training data, committed
  to this repository, or redistributed.
