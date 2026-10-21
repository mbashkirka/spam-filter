from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from spam_filter.api import create_app
from spam_filter.config import ProjectPaths
from spam_filter.data import sha256_file
from spam_filter.m4 import CalibratedSpamModel
from spam_filter.m8 import build_m8_serving_artifact
from spam_filter.m9 import RuntimeMetrics, drift_snapshot, log_prediction_event


def _seed_release(root: Path) -> ProjectPaths:
    paths = ProjectPaths.from_root(root)
    paths.ensure_runtime_dirs()
    texts = [
        'meeting at seven', 'see you tomorrow', 'free prize claim now',
        'win cash reward', 'call me later', 'urgent prize winner',
    ]
    labels = np.array([0, 0, 1, 1, 0, 1], dtype=np.int8)
    base = Pipeline([
        ('features', TfidfVectorizer(analyzer='char', ngram_range=(2, 4))),
        ('classifier', LinearSVC(random_state=42)),
    ]).fit(texts, labels)
    margins = np.asarray(base.decision_function(texts), dtype=float).reshape(-1, 1)
    calibrator = LogisticRegression(C=1000, max_iter=1000).fit(margins, labels)
    model = CalibratedSpamModel(
        base_model=base, calibrator=calibrator, threshold=0.8,
        fp_cost=10.0, fn_cost=1.0, selected_model_name='char_tfidf_linear_svc',
    )
    model_path = paths.artifacts / 'm4_calibrated_champion.joblib'
    joblib.dump(model, model_path)
    m4 = {
        'artifact': {'sha256': sha256_file(model_path)},
        'test_decision_metrics': {'precision': 1.0, 'recall': .9, 'f1': .947, 'fp': 0, 'fn': 1},
        'test_probability_metrics': {'pr_auc': .98, 'roc_auc': .99, 'brier': .01, 'ece': .02},
    }
    m6 = {'models': {
        'm4_calibrated_sparse': {'pr_auc': .33, 'f1': .27, 'probability': {'ece': .46}},
        'm5_transformer': {'pr_auc': .30, 'f1': .39, 'probability': {'ece': .69}},
    }}
    m7 = {'external': {'agreement': {'both_wrong': 2345}}, 'robustness': {'rows': []}}
    reports = [('m4_results.json', m4), ('m6_results.json', m6), ('m7_results.json', m7)]
    for name, payload in reports:
        (paths.reports / name).write_text(json.dumps(payload), encoding='utf-8')
    (paths.reports / 'm1_eda.json').write_text(json.dumps({
        'spam_rate': .1263, 'text_length_chars': {'median': 61.0, 'p95': 160.0}
    }), encoding='utf-8')
    build_m8_serving_artifact(paths)
    return paths


def test_runtime_metrics_contract() -> None:
    metrics = RuntimeMetrics()
    metrics.record_prediction(
        label='spam', probability=.9, input_chars=42, latency_ms=2.0, domain_warning=False
    )
    metrics.record_prediction(
        label='ham', probability=.1, input_chars=20, latency_ms=4.0, domain_warning=True
    )
    snapshot = metrics.snapshot()
    assert snapshot['requests_total'] == 2
    assert snapshot['predictions'] == {'spam': 1, 'ham': 1}
    assert snapshot['domain_warning_rate'] == pytest.approx(.5)
    assert 'spam_filter_requests_total 2' in metrics.prometheus_text()


def test_drift_requires_minimum_observations() -> None:
    metrics = RuntimeMetrics()
    result = drift_snapshot(metrics, {'spam_rate': .1, 'median_chars': 60, 'p95_chars': 160})
    assert result['status'] == 'insufficient_observations'


def test_drift_alerts_on_out_of_domain_rate() -> None:
    metrics = RuntimeMetrics()
    for _ in range(20):
        metrics.record_prediction(
            label='ham', probability=.1, input_chars=50, latency_ms=1, domain_warning=True
        )
    result = drift_snapshot(metrics, {'spam_rate': .1, 'median_chars': 60, 'p95_chars': 160})
    assert result['status'] == 'alert'
    assert result['signals']['out_of_domain_rate']['alert']


def test_structured_log_never_contains_raw_text(caplog) -> None:
    secret = 'THIS RAW MESSAGE MUST NEVER BE LOGGED'
    logger = logging.getLogger('test_privacy_logger')
    with caplog.at_level(logging.INFO, logger='test_privacy_logger'):
        log_prediction_event(
            logger, request_id='abc', channel='sms', input_chars=len(secret), label='ham',
            probability=.01, latency_ms=1.2, domain_warning=False, model_version='v1',
        )
    assert secret not in caplog.text
    payload = json.loads(caplog.records[-1].message)
    assert 'text' not in payload
    assert payload['input_chars'] == len(secret)


def test_m9_api_observability(tmp_path) -> None:
    _seed_release(tmp_path)
    client = TestClient(create_app(tmp_path))
    response = client.post('/predict', json={'text': 'free prize claim now', 'channel': 'sms'})
    assert response.status_code == 200
    assert response.json()['request_id']
    metrics = client.get('/metrics')
    assert metrics.status_code == 200
    assert 'spam_filter_requests_total 1' in metrics.text
    snap = client.get('/metrics.json').json()
    assert snap['requests_total'] == 1
    drift = client.get('/drift').json()
    assert drift['status'] == 'insufficient_observations'
    assert client.get('/health').json()['milestone'] == 'M9'


def test_m9_release_contract_end_to_end(tmp_path) -> None:
    from spam_filter.m8 import FrozenSpamService
    from spam_filter.m9 import (
        benchmark_service,
        build_release_contract,
        write_latency_report,
    )

    paths = _seed_release(tmp_path)
    for rel in ['MODEL_CARD.md', 'Dockerfile', '.dockerignore', '.github/workflows/ci.yml']:
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f'test fixture for {rel}\n', encoding='utf-8')
    (paths.reports / 'm8_report.md').write_text('# test m8 report\n', encoding='utf-8')
    service = FrozenSpamService.load(paths)
    latency = benchmark_service(service, iterations=20, warmup=2, slo_p95_ms=1000.0)
    write_latency_report(paths, latency)
    release = build_release_contract(paths, strict_python=False)
    assert release['release_version'] == '1.0.0'
    assert release['release_status'] == 'ready'
    assert release['privacy']['raw_text_logged'] is False
