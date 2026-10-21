# M5 Transformer benchmark contract

M5 introduces a compact pretrained Transformer benchmark without replacing the sparse M3/M4 production candidate.

## Default model

`distilbert-base-uncased` with a two-class sequence-classification head.

The default training path is intentionally small enough for a local Mac workflow:

- max sequence length: 128 tokens;
- batch size: 16;
- epoch budget: 3;
- AdamW learning rate: 2e-5;
- weighted cross-entropy from M1 train class frequencies;
- validation PR-AUC for checkpoint selection;
- patience: 1 epoch;
- Apple MPS preferred automatically, then CUDA, then CPU.

## Leakage contract

- tokenizer/model optimization sees M1 train only;
- checkpoint selection sees validation only;
- the already-reported M1 test split is used as a frozen benchmark;
- M6 adds an external email-domain benchmark before production claims.

## Artifacts

`artifacts/m5_transformer/` contains the Hugging Face-compatible local checkpoint plus a SHA-256 manifest.

Reports:

- `reports/m5_results.json`;
- `reports/m5_report.md`;
- `reports/m5_training_history.csv`.

## Why keep the sparse baseline?

Transformer complexity is not assumed to be beneficial. M5 reports the delta against the M3 char-TF-IDF + LinearSVC champion. The project keeps both model families until external-domain and robustness evidence justify a final choice.
