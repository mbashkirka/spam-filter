import numpy as np
import pandas as pd

from spam_filter.m3 import (
    M3Config,
    _continuous_scores,
    build_candidates,
    classifier_metrics,
    normalize_sparse_text,
)


def test_normalize_sparse_text_replaces_sensitive_patterns() -> None:
    text = normalize_sparse_text("MAIL A.B@test.org or visit https://x.com and pay 12.50")
    assert "emailtoken" in text
    assert "urltoken" in text
    assert "numtoken" in text
    assert "a.b@test.org" not in text


def test_build_candidates_contract() -> None:
    candidates = build_candidates(M3Config())
    assert set(candidates) == {
        "word_tfidf_mnb",
        "word_tfidf_logreg",
        "char_tfidf_logreg",
        "char_tfidf_linear_svc",
        "word_char_linear_svc",
    }


def test_candidates_fit_predict_and_score() -> None:
    frame = pd.DataFrame(
        {
            "text": [
                "see you at home tonight",
                "meeting moved to nine",
                "please call me later",
                "free prize claim now",
                "winner cash prize claim",
                "urgent win free reward",
                "lunch at office tomorrow",
                "guaranteed reward call now",
            ],
            "label": [0, 0, 0, 1, 1, 1, 0, 1],
        }
    )
    texts = frame["text"].tolist()
    labels = frame["label"].tolist()
    for model in build_candidates().values():
        model.fit(texts, labels)
        predictions = np.asarray(model.predict(texts))
        scores = _continuous_scores(model, texts)
        assert predictions.shape == (8,)
        assert scores.shape == (8,)
        assert np.isfinite(scores).all()


def test_classifier_metrics_confusion_contract() -> None:
    metrics = classifier_metrics(
        [0, 0, 1, 1],
        np.asarray([0, 1, 1, 0]),
        np.asarray([-2.0, 0.4, 2.0, -0.3]),
    )
    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tp"] == 1
