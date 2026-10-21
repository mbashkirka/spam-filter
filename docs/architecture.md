# Architecture direction

The project is intentionally layered so that each modeling decision can be benchmarked independently.

```text
raw text
  -> validation / de-duplication / split contract
  -> preprocessing variants
  -> feature/model families
       A. from-scratch multinomial NB
       B. sparse word/character TF-IDF + linear models
       C. compact transformer
  -> probability / score calibration
  -> business threshold policy
  -> frozen inference bundle
  -> API + metrics + drift monitoring
```

The main portfolio story is not "transformer beats Naive Bayes". It is the engineering and experimental path from an interpretable probabilistic baseline to a robust deployed NLP decision system.

## M7 diagnostic layer

The serving decision is intentionally deferred after M6 domain shift. M7 consumes the frozen M6 per-message audit and the frozen SMS test split, without fitting on either diagnostic set. It produces external length/content slices, M3-vs-M5 error agreement groups, deterministic obfuscation stress tests, and global LinearSVC character n-gram weights. These outputs become model-card and serving-domain evidence for M8/M9.
