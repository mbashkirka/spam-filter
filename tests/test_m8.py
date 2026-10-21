import json
from pathlib import Path

import joblib
import numpy as np
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from spam_filter.api import create_app
from spam_filter.config import ProjectPaths
from spam_filter.data import sha256_file
from spam_filter.m4 import CalibratedSpamModel
from spam_filter.m8 import (
    MODEL_VERSION,
    FrozenSpamService,
    build_m8_serving_artifact,
    verify_m8_artifact,
)


def _seed_prerequisites(root: Path) -> ProjectPaths:
    paths = ProjectPaths.from_root(root)
    paths.ensure_runtime_dirs()
    texts = [
        "meeting at seven",
        "see you tomorrow",
        "free prize claim now",
        "win cash reward",
        "call me later",
        "urgent prize winner",
    ]
    labels = np.array([0, 0, 1, 1, 0, 1], dtype=np.int8)
    base = Pipeline(
        [
            ("features", TfidfVectorizer(analyzer="char", ngram_range=(2, 4))),
            ("classifier", LinearSVC(random_state=42)),
        ]
    )
    base.fit(texts, labels)
    margins = np.asarray(base.decision_function(texts), dtype=float).reshape(-1, 1)
    calibrator = LogisticRegression(C=1000, max_iter=1000).fit(margins, labels)
    model = CalibratedSpamModel(
        base_model=base,
        calibrator=calibrator,
        threshold=0.8,
        fp_cost=10.0,
        fn_cost=1.0,
        selected_model_name="char_tfidf_linear_svc",
    )
    model_path = paths.artifacts / "m4_calibrated_champion.joblib"
    joblib.dump(model, model_path)
    m4 = {
        "artifact": {"sha256": sha256_file(model_path)},
        "test_decision_metrics": {
            "precision": 1.0,
            "recall": 0.9,
            "f1": 0.947,
            "fp": 0,
            "fn": 1,
        },
        "test_probability_metrics": {
            "pr_auc": 0.98,
            "roc_auc": 0.99,
            "brier": 0.01,
            "ece": 0.02,
        },
    }
    m6 = {
        "models": {
            "m4_calibrated_sparse": {
                "pr_auc": 0.33,
                "f1": 0.27,
                "probability": {"ece": 0.46},
            },
            "m5_transformer": {
                "pr_auc": 0.30,
                "f1": 0.39,
                "probability": {"ece": 0.69},
            },
        }
    }
    m7 = {
        "external": {"agreement": {"both_wrong": 2345}},
        "robustness": {
            "rows": [
                {
                    "attack": "clean",
                    "model": "m3_sparse",
                    "spam_detection_rate": 0.93,
                }
            ]
        },
    }
    reports = [("m4_results.json", m4), ("m6_results.json", m6), ("m7_results.json", m7)]
    for name, payload in reports:
        (paths.reports / name).write_text(json.dumps(payload), encoding="utf-8")
    return paths


def test_build_and_verify_m8(tmp_path) -> None:
    paths = _seed_prerequisites(tmp_path)
    payload = build_m8_serving_artifact(paths)
    assert payload["model_version"] == MODEL_VERSION
    assert payload["intended_use"]["channel"] == "sms"
    assert verify_m8_artifact(paths)
    service = FrozenSpamService.load(paths)
    result = service.predict_one("free prize claim now")
    assert 0.0 <= result["spam_probability"] <= 1.0


def test_m8_detects_tampered_model(tmp_path) -> None:
    paths = _seed_prerequisites(tmp_path)
    build_m8_serving_artifact(paths)
    model_path = paths.artifacts / "m8_serving" / "model.joblib"
    with model_path.open("ab") as handle:
        handle.write(b"tamper")
    assert not verify_m8_artifact(paths)


def test_api_contract(tmp_path) -> None:
    paths = _seed_prerequisites(tmp_path)
    build_m8_serving_artifact(paths)
    client = TestClient(create_app(tmp_path))
    assert client.get("/health").status_code == 200
    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    info = client.get("/model-info")
    assert info.status_code == 200
    assert info.json()["intended_use"]["channel"] == "sms"
    pred = client.post("/predict", json={"text": "free prize claim now", "channel": "sms"})
    assert pred.status_code == 200
    assert pred.json()["domain_warning"] is None
    email = client.post("/v1/predict", json={"text": "subject: prize", "channel": "email"})
    assert email.status_code == 200
    assert "Do not treat" in email.json()["domain_warning"]


def test_api_not_ready_without_artifact(tmp_path) -> None:
    client = TestClient(create_app(tmp_path))
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 503
