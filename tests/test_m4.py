import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from spam_filter.m4 import (
    CalibratedSpamModel,
    M4Config,
    expected_calibration_error,
    select_cost_threshold,
)


def test_m4_config_rejects_zero_costs() -> None:
    config = M4Config(fp_cost=0, fn_cost=0)
    try:
        config.validate()
    except ValueError as exc:
        assert "at least one" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_expected_calibration_error_is_zero_for_perfect_bins() -> None:
    y = np.array([0, 0, 1, 1], dtype=np.int8)
    prob = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(y, prob, bins=4) == 0.0


def test_cost_threshold_prefers_avoiding_expensive_false_positive() -> None:
    y = np.array([0, 1, 1], dtype=np.int8)
    prob = np.array([0.60, 0.70, 0.80])
    selected, _ = select_cost_threshold(y, prob, fp_cost=10.0, fn_cost=1.0)
    assert selected["fp"] == 0
    assert 0.60 < selected["threshold"] <= 0.70


def test_calibrated_spam_model_probability_and_threshold() -> None:
    texts = ["hello friend", "free prize now", "meeting tomorrow", "claim free cash"]
    labels = np.array([0, 1, 0, 1])
    base = Pipeline(
        [
            ("features", TfidfVectorizer()),
            ("classifier", LinearSVC()),
        ]
    )
    base.fit(texts, labels)
    margins = np.asarray(base.decision_function(texts), dtype=float)
    calibrator = LogisticRegression().fit(margins.reshape(-1, 1), labels)
    model = CalibratedSpamModel(
        base_model=base,
        calibrator=calibrator,
        threshold=0.5,
        fp_cost=10.0,
        fn_cost=1.0,
        selected_model_name="synthetic",
    )
    probabilities = model.predict_proba(["free prize", "meeting friend"])
    assert probabilities.shape == (2, 2)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert model.predict(["free prize", "meeting friend"]).shape == (2,)
