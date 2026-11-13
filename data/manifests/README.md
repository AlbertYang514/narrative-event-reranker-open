# Source Manifests

`*.jsonl` files in this directory record source metadata only. Raw text and
processed corpora are deliberately excluded from version control.

Fields:

- `source_id`: stable identifier used by scripts
- `title`, `author`: bibliographic metadata
- `source_url`: where the text can be obtained
- `license`: license / public-domain status
- `jurisdiction_checked`: whether the license status was checked
- `allowed_for_training`: whether the source may enter formal training data
- `notes`: import or processing caveats

A source can be recorded here without the repository redistributing its text.
Users must fetch the text themselves and comply with the source terms.

`sources.example.jsonl` is a template. `sources.public_domain.jsonl` records the
public-domain works used in the reported experiments.
