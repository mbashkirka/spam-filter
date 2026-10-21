# NLP Spam Detection Service

A production-oriented NLP pet project that starts from the classic Naive Bayes spam-classification idea and develops it into a versioned, tested, observable serving system.

**Final release: M9 / v1.0.0.**

## Portfolio summary

This repository demonstrates the full lifecycle of a practical NLP classifier rather than a single notebook:

| Milestone | Scope | Key outcome |
|---|---|---|
| M0 | Project contract | package, CLI, tests, linting |
| M1 | Data + EDA | UCI ingestion, deduplication, leakage-safe split |
| M2 | NLP fundamentals | scratch Multinomial Naive Bayes + preprocessing ablation |
| M3 | Sparse NLP | character TF-IDF + LinearSVC champion |
| M4 | Decision system | OOF Platt calibration + cost-sensitive threshold |
| M5 | Transformer | DistilBERT benchmark on Apple MPS |
| M6 | Domain shift | zero-shot AUEB Enron1 email benchmark |
| M7 | Robustness | slices, error taxonomy, evasion tests, explainability |
| M8 | Serving | frozen artifact + FastAPI production contract |
| **M9** | **Production polish** | **metrics, drift, privacy-safe logging, latency/SLO, CI, Docker, model card** |

## Key results

- **M2 scratch NB:** PR-AUC `0.9671`, F1 `0.9375`, FP/FN `4/8`.
- **M3 char-TF-IDF + LinearSVC:** PR-AUC `0.9860`, F1 `0.9630`, FP/FN `0/7`.
- **M4 calibrated serving policy:** threshold `0.866086`; precision `1.0000`, recall `0.8878`, Brier `0.0076`, ECE `0.0053` on the frozen SMS benchmark.
- **M5 DistilBERT:** PR-AUC `0.9909`, F1 `0.9694`, FP/FN `3/3`.
- **M6 external Enron1:** severe SMS→email domain shift; M4 PR-AUC `0.3334`, M5 PR-AUC `0.3011`.
- **M7:** confirms distinct sparse/Transformer robustness failure modes and that lexical/domain mismatch is not explained by message length alone.

The production default remains the calibrated sparse M4 model because it is compact, fast, calibrated, and has an explicit decision-cost policy. DistilBERT remains a challenger benchmark.

## Intended use

The frozen service is for **English SMS-like text only**. Email, long-form text, and non-English input are explicitly out of validated scope. The API exposes this limitation through `/model-info` and returns a domain warning for non-SMS channels.

See [`MODEL_CARD.md`](MODEL_CARD.md) for the complete model card.

## macOS setup

Python 3.12 is the validated runtime.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,data,transformer,serving]"
```

For serving/CI only, the lightweight stack is sufficient:

```bash
python -m pip install -e ".[dev,data,serving]"
```

## Final M9 verification

M9 expects the prior M8 frozen artifact and M6/M7 reports to remain in the repository runtime directories.

```bash
make m9-quality
make m9-test
make m6-smoke
make m7-smoke
make m8-smoke
make m9-benchmark
make m9-release
make m9-smoke
make m9-api-smoke
```

Or:

```bash
make m9-all
```

Generated final reports:

```text
reports/m9_latency.json
reports/m9_release.json
reports/m9_report.md
artifacts/m9_reference_profile.json
```

## API

Start locally:

```bash
make m9-serve
```

Endpoints:

- `GET /health`
- `GET /ready`
- `GET /model-info`
- `POST /predict`
- `POST /v1/predict`
- `GET /metrics`
- `GET /metrics.json`
- `GET /drift`

Example:

```bash
curl -X POST http://127.0.0.1:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"WIN a FREE prize now!","channel":"sms"}'
```

### Privacy-safe observability

Prediction logs do **not** contain the input text. The service records only a request id, declared channel, input length, output label, probability, latency, domain-warning flag, and model version.

`/metrics` contains process-local aggregate counters/gauges. `/drift` compares aggregate request behavior with the M1 reference profile after at least 20 observations. These are unsupervised signals and do not substitute for labeled production evaluation.

## Docker

Build only after `artifacts/m8_serving/` has been frozen:

```bash
docker build -t spam-filter-lab:1.0.0 .
docker run --rm -p 8000:8000 spam-filter-lab:1.0.0
```

The image excludes Transformer checkpoints and external evaluation datasets because the deployed M9 policy is the lightweight sparse model.

## CI

`.github/workflows/ci.yml` runs on Python 3.12 and executes Ruff plus the unit/contract test suite using the lightweight serving dependency set.

## Data provenance

Primary SMS data: UCI SMS Spam Collection, dataset id 228, DOI `10.24432/C5CC84`, CC BY 4.0. External transfer evidence: AUEB Enron-Spam / Enron1. Exact duplicate texts are removed before primary splitting/evaluation.

## Methodological principles

- exact-message deduplication before primary splitting;
- no cross-split text leakage;
- fit on train, select on validation;
- OOF calibration when probabilities become part of the decision system;
- explicit configurable FP/FN costs;
- PR-AUC and minority-class metrics over accuracy-only reporting;
- hashed/frozen model artifacts;
- external domain-shift evidence before deployment claims;
- no automatic adaptation on diagnostic external data;
- privacy-safe observability and explicit model boundaries.
