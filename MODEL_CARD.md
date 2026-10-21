# Model Card — Spam SMS Sparse Calibrated v1

## Model details

- **Serving model:** `spam-sms-sparse-calibrated-v1`
- **Project release:** `spam-filter-lab v1.0.0 / M9`
- **Architecture:** character TF-IDF + LinearSVC + out-of-fold Platt calibration
- **Decision policy:** predict spam when calibrated `P(spam)` is at or above the frozen M4 threshold
- **Primary language/domain:** English SMS-like messages

## Intended use

The model is intended as a portfolio/research-grade binary classifier for English SMS-like spam detection. It demonstrates a complete NLP lifecycle: leakage-safe data preparation, scratch Naive Bayes, sparse baselines, probability calibration, Transformer comparison, external domain-shift evaluation, robustness diagnostics, frozen serving, and observability.

It is **not validated for email, long-form documents, or non-English text**. API callers should send `channel="sms"` only when the request is actually SMS-like.

## Training and evaluation data

The primary training/evaluation dataset is the UCI SMS Spam Collection. Exact duplicate texts are removed before the deterministic train/validation/test split. The external AUEB Enron1 email corpus is used only as a zero-shot domain-shift benchmark and is not used to adapt the serving model.

## Frozen in-domain evidence

For the frozen SMS benchmark, the M4 serving policy achieved approximately:

- PR-AUC: `0.9860`
- F1: `0.9405`
- Precision: `1.0000`
- Recall: `0.8878`
- FP/FN: `0/11`
- Brier score: `0.0076`
- ECE: `0.0053`

M5 DistilBERT achieved higher model-quality metrics on the same frozen SMS benchmark (`PR-AUC 0.9909`, `F1 0.9694`) but is retained as a challenger rather than the production default because the sparse M4 policy is smaller, faster, calibrated, and has an explicit cost-sensitive decision rule.

## Domain-shift evidence

Zero-shot SMS→Enron1 transfer failed severely. M4 email PR-AUC/F1 were approximately `0.3334/0.2727`; M5 email PR-AUC/F1 were approximately `0.3011/0.3898`. Probability calibration also degraded substantially. This is treated as a confirmed out-of-domain risk, not as production evidence for email spam filtering.

## Robustness and limitations

M7 demonstrates different failure modes for sparse and Transformer models. Surface obfuscations such as inserted separators can reduce detection; Unicode confusables and domain-specific lexical changes also affect behavior. Character n-gram coefficients are global linear explanations, not causal explanations.

The serving threshold encodes an illustrative false-positive:false-negative cost ratio of `10:1`. Real deployment requires product-specific cost validation and fresh labeled monitoring.

## Privacy and observability

M9 structured logs and metrics do not store raw message text. Online process-level observability includes request count, channel, input length, prediction label, probability, latency, and domain-warning status. `/drift` exposes unsupervised aggregate signals only; it does not infer true accuracy without labels.

## Ethical and operational considerations

False positives can hide legitimate messages, while false negatives can expose users to unwanted or malicious spam. The model should not be used as the sole safety control for phishing, fraud, or security-sensitive message filtering. Human-visible quarantine/recovery mechanisms and labeled production evaluation are recommended for any real deployment.
