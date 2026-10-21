from pathlib import Path

import pytest

from spam_filter.naive_bayes import ScratchMultinomialNB, spam_probability
from spam_filter.preprocess import PreprocessConfig, TextPreprocessor


def _training_data() -> tuple[list[list[str]], list[int], TextPreprocessor]:
    preprocessor = TextPreprocessor(PreprocessConfig(name="test"))
    texts = [
        "meeting tomorrow project update",
        "lunch meeting with team",
        "project notes attached",
        "win cash prize now",
        "claim your free prize",
        "winner call now cash",
    ]
    return preprocessor.transform(texts), [0, 0, 0, 1, 1, 1], preprocessor


def test_scratch_nb_separates_simple_classes() -> None:
    tokens, labels, preprocessor = _training_data()
    model = ScratchMultinomialNB(alpha=1.0).fit(tokens, labels)
    ham = spam_probability(model, preprocessor, "team project meeting")
    spam = spam_probability(model, preprocessor, "win free cash prize")
    assert ham < 0.5
    assert spam > 0.5
    assert spam > ham


def test_probabilities_sum_to_one() -> None:
    tokens, labels, preprocessor = _training_data()
    model = ScratchMultinomialNB().fit(tokens, labels)
    probabilities = model.predict_proba(preprocessor.transform(["free prize", "team notes"]))
    assert probabilities.shape == (2, 2)
    assert probabilities.sum(axis=1).tolist() == pytest.approx([1.0, 1.0])


def test_model_roundtrip(tmp_path: Path) -> None:
    tokens, labels, preprocessor = _training_data()
    model = ScratchMultinomialNB().fit(tokens, labels)
    path = tmp_path / "model.json"
    model.save(path, preprocessor.config)
    restored, restored_preprocessor = ScratchMultinomialNB.load(path)
    expected = spam_probability(model, preprocessor, "win cash")
    actual = spam_probability(restored, restored_preprocessor, "win cash")
    assert actual == pytest.approx(expected)
