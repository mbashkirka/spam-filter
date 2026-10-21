import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from spam_filter.m7 import (
    M7Config,
    agreement_summary,
    attack_confusable,
    attack_leetspeak,
    attack_separator,
    attack_zero_width,
    enrich_external_frame,
    length_bucket,
    slice_metrics,
    sparse_explainability,
)


def test_m7_config_contract() -> None:
    M7Config().validate()
    with pytest.raises(ValueError):
        M7Config(batch_size=0).validate()
    with pytest.raises(ValueError):
        M7Config(device="tpu").validate()


def test_length_bucket_contract() -> None:
    assert length_bucket(160) == "sms_like_<=160"
    assert length_bucket(161) == "short_email_161_512"
    assert length_bucket(513) == "email_513_2048"
    assert length_bucket(2049) == "long_email_>2048"


def test_enrich_external_frame() -> None:
    frame = pd.DataFrame(
        {
            "text": ["Subject: offer http://x.com 123", "hello"],
            "label": [1, 0],
        }
    )
    enriched = enrich_external_frame(frame)
    assert bool(enriched.loc[0, "has_url_like"])
    assert bool(enriched.loc[0, "has_digits"])
    assert bool(enriched.loc[0, "has_subject_header"])
    assert enriched.loc[1, "length_bucket"] == "sms_like_<=160"


def test_slice_metrics_binary() -> None:
    y = np.asarray([0, 0, 1, 1], dtype=np.int8)
    pred = np.asarray([0, 1, 0, 1], dtype=np.int8)
    scores = np.asarray([0.1, 0.8, 0.4, 0.9])
    metrics = slice_metrics(y, pred, scores)
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["pr_auc"] is not None


def test_agreement_summary() -> None:
    frame = pd.DataFrame(
        {
            "label": [0, 0, 1, 1],
            "m3_prediction": [0, 1, 1, 0],
            "m5_prediction": [0, 0, 0, 0],
        }
    )
    summary = agreement_summary(frame)
    assert summary == {
        "both_correct": 1,
        "only_m3_correct": 1,
        "only_m5_correct": 1,
        "both_wrong": 1,
    }


def test_attacks_change_long_words() -> None:
    text = "Congratulations winner claim prize immediately"
    for attack in [attack_leetspeak, attack_separator, attack_zero_width, attack_confusable]:
        attacked = attack(text)
        assert attacked != text
        assert len(attacked) >= len(text)


def test_sparse_explainability() -> None:
    model = Pipeline(
        [
            ("features", TfidfVectorizer(analyzer="char", ngram_range=(2, 3))),
            ("classifier", LinearSVC()),
        ]
    )
    model.fit(["normal meeting", "project update", "claim prize", "free winner"], [0, 0, 1, 1])
    table = sparse_explainability(model, 5)
    assert len(table) == 10
    assert set(table["direction"]) == {"spam", "ham"}


def test_run_m7_synthetic_contract(tmp_path, monkeypatch) -> None:
    import hashlib

    import joblib

    import spam_filter.m7 as m7_module
    from spam_filter.config import ProjectPaths

    paths = ProjectPaths.from_root(tmp_path)
    paths.ensure_runtime_dirs()

    texts = [
        "Subject: team meeting tomorrow",
        "claim your prize now http://offer.com",
        "project update and budget notes",
        "free winner cash offer immediately",
    ]
    labels = [0, 1, 0, 1]
    hashes = [hashlib.sha256(text.encode()).hexdigest() for text in texts]
    external = pd.DataFrame(
        {
            "source": ["enron1"] * 4,
            "file_name": [f"{i}.txt" for i in range(4)],
            "text": texts,
            "label_name": ["ham", "spam", "ham", "spam"],
            "label": labels,
            "text_sha256": hashes,
        }
    )
    external.to_csv(paths.data_processed / "enron1_external.csv", index=False)

    predictions = external[["source", "file_name", "label_name", "label", "text_sha256"]].copy()
    predictions["m3_prediction"] = [0, 1, 0, 1]
    predictions["m3_score"] = [-1.0, 1.0, -0.5, 0.8]
    predictions["m4_probability"] = [0.01, 0.9, 0.02, 0.85]
    predictions["m4_prediction"] = [0, 1, 0, 0]
    predictions["m5_probability"] = [0.1, 0.8, 0.2, 0.7]
    predictions["m5_prediction"] = [0, 1, 0, 1]
    predictions.to_csv(paths.reports / "m6_external_predictions.csv", index=False)

    test = pd.DataFrame(
        {
            "text": ["normal hello", "claim prize winner", "free cash offer"],
            "label": [0, 1, 1],
        }
    )
    test.to_csv(paths.data_processed / "test.csv", index=False)

    model = Pipeline(
        [
            ("features", TfidfVectorizer(analyzer="char", ngram_range=(2, 3))),
            ("classifier", LinearSVC()),
        ]
    )
    model.fit(["hello meeting", "project notes", "claim prize", "free cash winner"], [0, 0, 1, 1])
    joblib.dump(model, paths.artifacts / "m3_sparse_champion.joblib")
    m5_dir = paths.artifacts / "m5_transformer"
    m5_dir.mkdir(parents=True)
    (m5_dir / "m5_manifest.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        m7_module,
        "load_m5_for_inference",
        lambda *args, **kwargs: (None, None, "cpu"),
    )
    monkeypatch.setattr(
        m7_module,
        "_predict_transformer_batches",
        lambda texts, *args, **kwargs: np.full(len(texts), 0.9, dtype=float),
    )

    result = m7_module.run_m7(paths, M7Config(batch_size=2, device="cpu", top_features=5))
    assert result["external"]["rows"] == 4
    assert result["external"]["agreement"]["both_wrong"] == 0
    assert result["robustness"]["spam_rows"] == 2
    assert (paths.reports / "m7_report.md").exists()
    assert all(m7_module.verify_m7_outputs(paths).values())
