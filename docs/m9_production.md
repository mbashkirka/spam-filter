# M9 production contract

M9 promotes the project to `v1.0.0` without changing the frozen M4 serving decision model.

## Runtime endpoints

- `GET /health` — process liveness.
- `GET /ready` — verified model artifact readiness.
- `GET /model-info` — model provenance, intended use, and domain-risk evidence.
- `POST /predict` and `POST /v1/predict` — prediction contract.
- `GET /metrics` — process-local Prometheus text metrics.
- `GET /metrics.json` — JSON metric snapshot.
- `GET /drift` — unsupervised aggregate drift indicators.

## Privacy contract

Raw message text is never emitted to structured prediction logs or observability metrics. The online log event contains only request id, declared channel, message length, predicted label, probability, latency, domain-warning flag, and model version.

## Drift contract

Drift monitoring is intentionally conservative and label-free. After at least 20 observations it checks:

1. non-SMS/domain-warning rate;
2. mean input length relative to the M1 p95 length;
3. absolute shift in predicted spam rate relative to the M1 spam prevalence.

An alert is diagnostic only. It does not trigger automatic retraining or threshold changes.

## Release contract

`make m9-release` requires a valid M8 frozen artifact, completed M6/M7 evidence, the M9 latency benchmark, Model Card, Dockerfile, dockerignore, and CI workflow. It writes a hashed release report under `reports/`.

## Deployment contract

The Docker image contains the sparse serving stack only. Heavy Transformer artifacts and external evaluation datasets are excluded from the image context. The model artifact under `artifacts/m8_serving/` must exist before the image is built.
