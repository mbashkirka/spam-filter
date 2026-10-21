# M1 data contract

## Objective

Create a deterministic, leakage-aware SMS spam benchmark before any NLP feature engineering or model training.

## Canonical schema

| column | meaning |
|---|---|
| `source_row_id` | row position in the fetched UCI snapshot |
| `text` | raw SMS text; no NLP normalization is applied in M1 |
| `label_name` | canonical string label: `ham` or `spam` |
| `label` | integer target: `ham=0`, `spam=1` |
| `text_sha256` | SHA-256 of the exact raw text |

## Leakage policy

1. Missing and empty messages are rejected.
2. Unknown labels are rejected.
3. A message text appearing with conflicting labels aborts preparation.
4. Exact duplicate message texts are removed **before** splitting.
5. The unique-message corpus is split 70/15/15 with stratification and seed 42.
6. SHA-256 sets for train/validation/test must be pairwise disjoint.
7. Generated files are hashed in `m1_manifest.json`.

M1 intentionally does not lowercase, tokenize, stem, lemmatize, remove stop words, or mask URLs/numbers. Those are model/preprocessing decisions reserved for M2, so the raw benchmark contract stays auditable.
