from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from spam_filter.preprocess import PreprocessConfig, TextPreprocessor


@dataclass(frozen=True, slots=True)
class Prediction:
    label: int
    spam_probability: float


class ScratchMultinomialNB:
    """Binary multinomial Naive Bayes implemented without sklearn estimators/vectorizers."""

    def __init__(self, alpha: float = 1.0, min_df: int = 1):
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        if min_df < 1:
            raise ValueError("min_df must be >= 1")
        self.alpha = float(alpha)
        self.min_df = int(min_df)
        self.vocabulary_: dict[str, int] | None = None
        self.class_log_prior_: np.ndarray | None = None
        self.feature_log_prob_: np.ndarray | None = None
        self.class_count_: np.ndarray | None = None

    def fit(
        self, tokenized_texts: Sequence[Sequence[str]], y: Sequence[int]
    ) -> ScratchMultinomialNB:
        if len(tokenized_texts) != len(y) or not y:
            raise ValueError("tokenized_texts and y must be non-empty and have equal length")
        labels = np.asarray(y, dtype=np.int8)
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError("ScratchMultinomialNB requires both binary labels {0, 1}")

        document_frequency: Counter[str] = Counter()
        for tokens in tokenized_texts:
            document_frequency.update(set(tokens))
        vocabulary_tokens = sorted(
            token for token, df in document_frequency.items() if df >= self.min_df
        )
        if not vocabulary_tokens:
            raise ValueError("Vocabulary is empty after min_df filtering")
        self.vocabulary_ = {token: index for index, token in enumerate(vocabulary_tokens)}

        class_count = np.bincount(labels, minlength=2).astype(float)
        feature_count = np.zeros((2, len(self.vocabulary_)), dtype=np.float64)
        for tokens, label in zip(tokenized_texts, labels, strict=True):
            counts = Counter(tokens)
            for token, count in counts.items():
                index = self.vocabulary_.get(token)
                if index is not None:
                    feature_count[label, index] += count

        self.class_count_ = class_count
        self.class_log_prior_ = np.log(class_count / class_count.sum())
        smoothed = feature_count + self.alpha
        self.feature_log_prob_ = np.log(smoothed / smoothed.sum(axis=1, keepdims=True))
        return self

    def _require_fitted(self) -> tuple[dict[str, int], np.ndarray, np.ndarray]:
        if (
            self.vocabulary_ is None
            or self.class_log_prior_ is None
            or self.feature_log_prob_ is None
        ):
            raise RuntimeError("Model is not fitted")
        return self.vocabulary_, self.class_log_prior_, self.feature_log_prob_

    def predict_log_proba(self, tokenized_texts: Sequence[Sequence[str]]) -> np.ndarray:
        vocabulary, class_log_prior, feature_log_prob = self._require_fitted()
        scores = np.tile(class_log_prior, (len(tokenized_texts), 1)).astype(np.float64)
        for row_index, tokens in enumerate(tokenized_texts):
            for token, count in Counter(tokens).items():
                feature_index = vocabulary.get(token)
                if feature_index is not None:
                    scores[row_index] += count * feature_log_prob[:, feature_index]
        max_scores = scores.max(axis=1, keepdims=True)
        log_norm = max_scores + np.log(np.exp(scores - max_scores).sum(axis=1, keepdims=True))
        return scores - log_norm

    def predict_proba(self, tokenized_texts: Sequence[Sequence[str]]) -> np.ndarray:
        return np.exp(self.predict_log_proba(tokenized_texts))

    def predict(self, tokenized_texts: Sequence[Sequence[str]]) -> np.ndarray:
        return self.predict_proba(tokenized_texts).argmax(axis=1).astype(np.int8)

    def top_tokens(self, class_label: int, n: int = 20) -> list[tuple[str, float]]:
        vocabulary, _, feature_log_prob = self._require_fitted()
        if class_label not in (0, 1):
            raise ValueError("class_label must be 0 or 1")
        inverse = [None] * len(vocabulary)
        for token, index in vocabulary.items():
            inverse[index] = token
        other = 1 - class_label
        log_odds = feature_log_prob[class_label] - feature_log_prob[other]
        indices = np.argsort(log_odds)[::-1][:n]
        return [(str(inverse[index]), float(log_odds[index])) for index in indices]

    def to_dict(self, preprocess_config: PreprocessConfig) -> dict[str, object]:
        vocabulary, class_log_prior, feature_log_prob = self._require_fitted()
        return {
            "format": "spam-filter-scratch-mnb-v1",
            "alpha": self.alpha,
            "min_df": self.min_df,
            "preprocess": preprocess_config.to_dict(),
            "vocabulary": vocabulary,
            "class_log_prior": class_log_prior.tolist(),
            "feature_log_prob": feature_log_prob.tolist(),
            "class_count": None if self.class_count_ is None else self.class_count_.tolist(),
        }

    def save(self, path: Path, preprocess_config: PreprocessConfig) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(preprocess_config), separators=(",", ":"))
        path.write_text(payload, encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> tuple[ScratchMultinomialNB, TextPreprocessor]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != "spam-filter-scratch-mnb-v1":
            raise ValueError("Unsupported model artifact format")
        config = PreprocessConfig(**payload["preprocess"])
        model = cls(alpha=float(payload["alpha"]), min_df=int(payload["min_df"]))
        model.vocabulary_ = {
            str(token): int(index) for token, index in payload["vocabulary"].items()
        }
        model.class_log_prior_ = np.asarray(payload["class_log_prior"], dtype=np.float64)
        model.feature_log_prob_ = np.asarray(payload["feature_log_prob"], dtype=np.float64)
        class_count = payload.get("class_count")
        model.class_count_ = (
            None if class_count is None else np.asarray(class_count, dtype=np.float64)
        )
        return model, TextPreprocessor(config)


def spam_probability(
    model: ScratchMultinomialNB, preprocessor: TextPreprocessor, text: str
) -> float:
    probability = float(model.predict_proba([preprocessor.tokenize(text)])[0, 1])
    if not math.isfinite(probability):
        raise RuntimeError("Model produced a non-finite probability")
    return probability
