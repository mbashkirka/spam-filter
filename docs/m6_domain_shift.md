# M6 — External Email Domain Shift

M6 evaluates the frozen SMS-trained models on the external **Enron-Spam / Enron1** email corpus published by the Athens University of Economics and Business (AUEB).

## Protocol

1. Download the official preprocessed `enron1.tar.gz` archive.
2. Parse the `ham/` and `spam/` messages without extracting untrusted archive paths.
3. Hash exact message text and remove duplicate exact texts for evaluation weighting.
4. Do **not** fit, calibrate, select, or tune on Enron labels.
5. Evaluate three frozen systems zero-shot:
   - M3 char-TF-IDF + LinearSVC;
   - M4 calibrated sparse classifier at its frozen SMS-validation threshold;
   - M5 DistilBERT at probability threshold 0.5.
6. Save per-message predictions for later M7 error/slice analysis.

## Why this milestone exists

The M1–M5 benchmark reuses the same SMS domain. M6 measures domain-transfer risk by changing both message format and language distribution from short SMS to real email. External performance is therefore more informative for portfolio claims than another score on the original split.

## Reproducibility

The raw archive, processed CSV, and prediction audit file are SHA-256 fingerprinted. The expected raw Enron1 class counts are 3,672 ham and 1,500 spam; a count mismatch fails preparation by default so source changes are visible rather than silently accepted.
