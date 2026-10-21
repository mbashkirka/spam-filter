# M2 NLP fundamentals contract

M2 intentionally implements the classifier rather than delegating it to `sklearn.naive_bayes`.

## Experimental protocol

- Fit all token statistics and model parameters on `train.csv` only.
- Compare four preprocessing variants on `validation.csv` only.
- Select the variant by validation PR-AUC, with F1 as a deterministic tie-breaker.
- Evaluate the selected configuration on `test.csv` exactly once.
- Keep the decision threshold fixed at `0.5` in M2. Threshold optimization is deferred to M4.

## Preprocessing ablation

1. `minimal`: Unicode normalization, lowercase, lexical tokenization.
2. `normalized_placeholders`: additionally maps URLs, e-mails and numbers to semantic placeholders.
3. `normalized_stopwords`: additionally removes English stop words.
4. `normalized_stopwords_stem`: additionally applies Porter stemming.

## Model

`ScratchMultinomialNB` computes class priors and Laplace-smoothed token likelihoods directly from token counts. It uses log-space scoring for numerical stability and serializes to a transparent JSON artifact.

## Primary metrics

PR-AUC is the selection metric because spam is the minority class. Precision, recall, F1, ROC-AUC, false positives and false negatives are also reported. Accuracy is retained only as a secondary diagnostic.
