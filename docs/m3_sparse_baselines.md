# M3 Sparse NLP Benchmark Contract

M3 benchmarks strong CPU-friendly sparse-text models against the transparent M2 scratch Naive Bayes baseline.

## Frozen inputs

M3 reuses the exact M1 `train.csv`, `validation.csv`, and `test.csv` files. Their SHA-256 hashes are recorded in the M3 result artifact.

## Normalization

M2 showed that placeholder normalization outperformed stopword removal and stemming. M3 therefore preserves that result: Unicode normalization + lowercase + URL/e-mail/number placeholders, with no stopword removal and no stemming.

## Candidate models

1. word (1,2)-gram TF-IDF + MultinomialNB;
2. word (1,2)-gram TF-IDF + LogisticRegression;
3. character-within-word (3,5)-gram TF-IDF + LogisticRegression;
4. character-within-word (3,5)-gram TF-IDF + LinearSVC;
5. word + character TF-IDF FeatureUnion + LinearSVC.

All vectorizers are fit on training data only. All candidates are selected on validation PR-AUC, with F1 as tie-breaker. Only the selected M3 model is evaluated on the frozen test split.

## Why character n-grams

Spam frequently contains unusual morphology, deliberate misspellings, concatenation, phone/URL fragments, and lexical variants. Character n-grams give the classical NLP baseline some robustness to these effects without requiring a transformer.

## Outputs

- `artifacts/m3_sparse_champion.joblib`
- `reports/m3_results.json`
- `reports/m3_report.md`

M3 scores from LinearSVC are ranking margins rather than probabilities. Calibration and cost-aware operating-threshold selection are intentionally deferred to M4.
