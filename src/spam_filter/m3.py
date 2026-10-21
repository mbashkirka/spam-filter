from __future__ import annotations

import json
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from spam_filter.config import ProjectPaths
from spam_filter.data import sha256_file
from spam_filter.preprocess import EMAIL_RE, NUMBER_RE, URL_RE


@dataclass(frozen=True, slots=True)
class M3Config:
    selection_metric: str = "pr_auc"
    random_state: int = 42


def normalize_sparse_text(text: str) -> str:
    """Apply the M2-winning lightweight normalization without stopword removal/stemming."""
    value = unicodedata.normalize("NFKC", str(text)).lower()
    value = EMAIL_RE.sub(" emailtoken ", value)
    value = URL_RE.sub(" urltoken ", value)
    value = NUMBER_RE.sub(" numtoken ", value)
    return value


def _word_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        preprocessor=normalize_sparse_text,
        lowercase=False,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.995,
        sublinear_tf=True,
        max_features=30_000,
    )


def _char_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        preprocessor=normalize_sparse_text,
        lowercase=False,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        max_features=40_000,
    )


def build_candidates(config: M3Config | None = None) -> dict[str, Pipeline]:
    config = config or M3Config()
    word_char = FeatureUnion(
        [
            ("word", _word_vectorizer()),
            ("char", _char_vectorizer()),
        ]
    )
    return {
        "word_tfidf_mnb": Pipeline(
            [("features", _word_vectorizer()), ("classifier", MultinomialNB(alpha=0.5))]
        ),
        "word_tfidf_logreg": Pipeline(
            [
                ("features", _word_vectorizer()),
                (
                    "classifier",
                    LogisticRegression(
                        C=2.0,
                        max_iter=2_000,
                        solver="liblinear",
                        random_state=config.random_state,
                    ),
                ),
            ]
        ),
        "char_tfidf_logreg": Pipeline(
            [
                ("features", _char_vectorizer()),
                (
                    "classifier",
                    LogisticRegression(
                        C=2.0,
                        max_iter=2_000,
                        solver="liblinear",
                        random_state=config.random_state,
                    ),
                ),
            ]
        ),
        "char_tfidf_linear_svc": Pipeline(
            [("features", _char_vectorizer()), ("classifier", LinearSVC(C=1.0))]
        ),
        "word_char_linear_svc": Pipeline(
            [("features", word_char), ("classifier", LinearSVC(C=1.0))]
        ),
    }


def _load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required M1 split not found: {path}. Run `make m1-data` first.")
    frame = pd.read_csv(path)
    required = {"text", "label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Split {path} is missing columns: {sorted(missing)}")
    return frame


def _continuous_scores(model: BaseEstimator, texts: list[str]) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(texts)  # type: ignore[attr-defined]
        return np.asarray(probabilities, dtype=float)[:, 1]
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(texts), dtype=float)  # type: ignore[attr-defined]
    raise TypeError("Estimator must implement predict_proba or decision_function")


def classifier_metrics(
    y_true: list[int], predictions: np.ndarray, scores: np.ndarray
) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _feature_count(model: Pipeline) -> int:
    features = model.named_steps["features"]
    if hasattr(features, "get_feature_names_out"):
        return int(len(features.get_feature_names_out()))
    return -1


def _evaluate_candidate(
    name: str,
    model: Pipeline,
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> dict[str, Any]:
    train_texts = train["text"].astype(str).tolist()
    train_labels = train["label"].astype(int).tolist()
    validation_texts = validation["text"].astype(str).tolist()
    validation_labels = validation["label"].astype(int).tolist()

    model.fit(train_texts, train_labels)
    predictions = np.asarray(model.predict(validation_texts), dtype=np.int8)
    scores = _continuous_scores(model, validation_texts)
    metrics = classifier_metrics(validation_labels, predictions, scores)
    metrics["name"] = name
    metrics["features"] = _feature_count(model)
    return metrics


def _m2_baseline(paths: ProjectPaths) -> dict[str, Any] | None:
    report_path = paths.reports / "m2_results.json"
    if not report_path.exists():
        return None
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        "name": "m2_scratch_nb",
        "validation": payload.get("selected"),
        "test": payload.get("test_metrics"),
    }


def _write_report(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# M3 Strong Sparse NLP Baselines",
        "",
        "All M3 candidate models are fit on the frozen M1 train split and selected using "
        "validation PR-AUC only. The selected M3 champion is then evaluated on the "
        "frozen test split.",
        "",
        "M3 deliberately keeps the M2-winning lightweight placeholder normalization and does not "
        "remove stopwords or apply stemming.",
        "",
        "## Validation benchmark",
        "",
        "| model | features | precision | recall | F1 | PR-AUC | ROC-AUC | FP | FN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in payload["validation_results"]:
        lines.append(
            f"| {item['name']} | {item['features']} | {item['precision']:.4f} | "
            f"{item['recall']:.4f} | {item['f1']:.4f} | {item['pr_auc']:.4f} | "
            f"{item['roc_auc']:.4f} | {item['fp']} | {item['fn']} |"
        )

    selected = payload["selected_validation"]
    test = payload["test_metrics"]
    lines.extend(
        [
            "",
            "## Selected M3 champion",
            "",
            f"- Model: `{payload['selected_model']}`",
            f"- Selection metric: `{payload['config']['selection_metric']}`",
            f"- Validation PR-AUC: {selected['pr_auc']:.4f}",
            f"- Validation F1: {selected['f1']:.4f}",
            "",
            "## Frozen test benchmark",
            "",
            f"- Precision: {test['precision']:.4f}",
            f"- Recall: {test['recall']:.4f}",
            f"- F1: {test['f1']:.4f}",
            f"- PR-AUC: {test['pr_auc']:.4f}",
            f"- ROC-AUC: {test['roc_auc']:.4f}",
            f"- Confusion matrix: TN={test['tn']}, FP={test['fp']}, "
            f"FN={test['fn']}, TP={test['tp']}",
        ]
    )

    baseline = payload.get("m2_baseline")
    if baseline and baseline.get("test"):
        m2_test = baseline["test"]
        lines.extend(
            [
                "",
                "## Comparison with M2 scratch Naive Bayes",
                "",
                f"- M2 test PR-AUC: {m2_test['pr_auc']:.4f}",
                f"- M3 test PR-AUC: {test['pr_auc']:.4f}",
                f"- PR-AUC delta: {test['pr_auc'] - m2_test['pr_auc']:+.4f}",
                f"- M2 test F1: {m2_test['f1']:.4f}",
                f"- M3 test F1: {test['f1']:.4f}",
                f"- F1 delta: {test['f1'] - m2_test['f1']:+.4f}",
                f"- M2 FP/FN: {m2_test['fp']}/{m2_test['fn']}",
                f"- M3 FP/FN: {test['fp']}/{test['fn']}",
            ]
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- PR-AUC is the primary ranking metric because spam is the minority class.",
            "- LinearSVC ranking scores are decision-function margins, not calibrated "
            "probabilities.",
            "- Probability calibration and business-cost threshold selection are intentionally "
            "deferred to M4.",
            "- Character n-grams are included because they can capture spelling variation and "
            "partial obfuscation better than word-only features.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_m3(paths: ProjectPaths, config: M3Config | None = None) -> dict[str, Any]:
    config = config or M3Config()
    if config.selection_metric not in {"pr_auc", "f1", "precision", "recall", "roc_auc"}:
        raise ValueError(f"Unsupported selection_metric: {config.selection_metric}")
    paths.ensure_runtime_dirs()

    train_path = paths.data_processed / "train.csv"
    validation_path = paths.data_processed / "validation.csv"
    test_path = paths.data_processed / "test.csv"
    train = _load_split(train_path)
    validation = _load_split(validation_path)
    test = _load_split(test_path)

    candidates = build_candidates(config)
    validation_results: list[dict[str, Any]] = []
    for name, model in candidates.items():
        validation_results.append(_evaluate_candidate(name, model, train, validation))
    validation_results.sort(
        key=lambda item: (item[config.selection_metric], item["f1"]), reverse=True
    )

    selected_name = validation_results[0]["name"]
    selected_validation = validation_results[0]
    selected_model = candidates[selected_name]

    test_texts = test["text"].astype(str).tolist()
    test_labels = test["label"].astype(int).tolist()
    test_predictions = np.asarray(selected_model.predict(test_texts), dtype=np.int8)
    test_scores = _continuous_scores(selected_model, test_texts)
    test_metrics = classifier_metrics(test_labels, test_predictions, test_scores)

    model_path = paths.artifacts / "m3_sparse_champion.joblib"
    joblib.dump(selected_model, model_path, compress=3)

    payload: dict[str, Any] = {
        "milestone": "M3",
        "config": asdict(config),
        "selection_protocol": (
            "fit train; select on validation; evaluate selected M3 champion on frozen test"
        ),
        "normalization": "M2 normalized_placeholders semantics; no stopword removal; no stemming",
        "data_sha256": {
            "train": sha256_file(train_path),
            "validation": sha256_file(validation_path),
            "test": sha256_file(test_path),
        },
        "validation_results": validation_results,
        "selected_model": selected_name,
        "selected_validation": selected_validation,
        "test_metrics": test_metrics,
        "m2_baseline": _m2_baseline(paths),
        "artifact": {
            "path": str(model_path.relative_to(paths.root)),
            "sha256": sha256_file(model_path),
            "bytes": model_path.stat().st_size,
        },
    }

    json_path = paths.reports / "m3_results.json"
    markdown_path = paths.reports / "m3_report.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(payload, markdown_path)
    payload["model_path"] = str(model_path)
    payload["report_json"] = str(json_path)
    payload["report_markdown"] = str(markdown_path)
    return payload


def load_m3_model(path: Path) -> Pipeline:
    model = joblib.load(path)
    if not isinstance(model, Pipeline):
        raise TypeError("M3 artifact is not a sklearn Pipeline")
    return model
