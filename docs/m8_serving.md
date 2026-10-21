# M8 Serving Contract

M8 freezes the M4 calibrated sparse policy as the production-oriented default serving artifact. M5 DistilBERT remains a benchmark rather than the default runtime model.

## Why M4 is served

- probability calibration was learned leakage-safely from out-of-fold train scores;
- the operating threshold was selected on validation under an explicit FP/FN cost assumption;
- the sparse model is much smaller and cheaper to serve than DistilBERT;
- M6/M7 show that both sparse and Transformer models fail under SMS→email domain shift, so M8 must restrict intended use rather than claim general spam detection.

## API

- `GET /health` — process-level liveness; does not require the model artifact;
- `GET /ready` — verifies and loads the frozen artifact;
- `GET /model-info` — model/version/threshold/evidence/intended-use/domain-risk metadata;
- `POST /predict` and `POST /v1/predict` — one-message classification.

Request:

```json
{"text": "WIN a free prize now", "channel": "sms"}
```

The API never echoes the input text in its response.

## Domain contract

Intended use is English SMS-like text. Email and long-form documents are explicitly not validated. M6 Enron1 evidence and M7 diagnostics are embedded into the serving manifest so downstream systems can inspect known domain risk.

## Artifact integrity

`artifacts/m8_serving/` contains a copied frozen M4 model and `manifest.json`. The manifest records the model SHA-256, source-report hashes, threshold, benchmark evidence, and a canonical artifact fingerprint. `/ready` fails if verification fails.
