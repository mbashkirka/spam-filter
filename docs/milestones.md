# Milestones

- **M0 — Repository contract:** package, CLI, tests, quality/smoke targets.
- **M1 — Data & EDA:** UCI SMS Spam Collection ingestion, schema validation, exact-deduplication, leakage-safe 70/15/15 split, SHA-256 manifest, EDA report.
- **M2 — NLP fundamentals:** preprocessing ablations and Multinomial Naive Bayes implemented from scratch. **Implemented in v0.3.0.**
- **M3 — Sparse baselines:** word/character TF-IDF + NB, Logistic Regression and LinearSVC. **Implemented in v0.4.0.**
- **M4 — Decision system:** OOF Platt calibration and configurable false-positive-aware threshold selection. **Implemented in v0.5.0.**
- **M5 — Transformer:** pretrained DistilBERT sequence-classification benchmark with MPS/CUDA/CPU fallback, validation checkpoint selection, hashed local checkpoint. **Implemented in v0.6.0.**
- **M6 — Email transfer:** official AUEB Enron1 ingestion and zero-shot domain-shift analysis. **Implemented in v0.7.1 (TLS downloader hotfix).**
- **M7 — Robustness:** obfuscated spam, slices, explainability and failure taxonomy. **Implemented in v0.8.0.**
- **M8 — Serving:** frozen artifact and FastAPI inference contract. **Implemented in v0.9.0.**
- **M9 — Production polish:** metrics, drift, CI/release contract, model card and demo.

## M6 — External Enron email domain shift

- Official AUEB Enron-Spam / Enron1 ingestion with archive fingerprinting.
- External zero-shot comparison of frozen M3, M4, and M5 systems.
- No fitting or threshold selection on Enron labels.
- Per-message prediction audit for M7 error analysis.

## M7 — Robustness and failure analysis

- External Enron error slices and M3-vs-M5 agreement taxonomy.
- Deterministic SMS spam evasion stress tests with no adversarial training.
- Global LinearSVC character n-gram explainability export.
- Hash-checked diagnostic CSV artifacts for downstream serving/model-card work.

## M8 — Production serving

Freeze the calibrated sparse SMS policy into a hashed serving artifact and expose health, readiness, model metadata, and prediction endpoints through FastAPI. Domain-risk evidence from M6/M7 is part of the serving contract.
