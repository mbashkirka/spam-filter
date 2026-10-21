from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from spam_filter.config import ProjectPaths
from spam_filter.data import sha256_file
from spam_filter.naive_bayes import ScratchMultinomialNB
from spam_filter.preprocess import PREPROCESSING_VARIANTS, PreprocessConfig, TextPreprocessor


@dataclass(frozen=True, slots=True)
class M2Config:
    alpha: float = 1.0
    min_df: int = 1
    selection_metric: str = "pr_auc"


def _load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required M1 split not found: {path}. Run `make m1-data` first.")
    frame = pd.read_csv(path)
    required = {"text", "label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Split {path} is missing columns: {sorted(missing)}")
    return frame


def binary_metrics(y_true: list[int], probabilities: list[float]) -> dict[str, Any]:
    predictions = [int(probability >= 0.5) for probability in probabilities]
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _evaluate_variant(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    variant: PreprocessConfig,
    config: M2Config,
) -> tuple[ScratchMultinomialNB, TextPreprocessor, dict[str, Any]]:
    preprocessor = TextPreprocessor(variant)
    train_tokens = preprocessor.transform(train["text"].astype(str))
    validation_tokens = preprocessor.transform(validation["text"].astype(str))
    model = ScratchMultinomialNB(alpha=config.alpha, min_df=config.min_df).fit(
        train_tokens, train["label"].astype(int).tolist()
    )
    probabilities = model.predict_proba(validation_tokens)[:, 1].tolist()
    metrics = binary_metrics(validation["label"].astype(int).tolist(), probabilities)
    metrics["variant"] = variant.name
    metrics["vocabulary_size"] = len(model.vocabulary_ or {})
    return model, preprocessor, metrics


def _write_report(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# M2 NLP Fundamentals Report",
        "",
        "Selection uses the validation split only. The test split is evaluated once "
        "after the preprocessing variant is selected.",
        "",
        "## Validation preprocessing ablation",
        "",
        "| variant | vocab | precision | recall | F1 | PR-AUC | ROC-AUC | FP | FN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in payload["validation_results"]:
        lines.append(
            f"| {item['variant']} | {item['vocabulary_size']} | {item['precision']:.4f} | "
            f"{item['recall']:.4f} | {item['f1']:.4f} | {item['pr_auc']:.4f} | "
            f"{item['roc_auc']:.4f} | {item['fp']} | {item['fn']} |"
        )
    selected = payload["selected"]
    test = payload["test_metrics"]
    lines.extend(
        [
            "",
            "## Selected configuration",
            "",
            f"- Variant: `{selected['variant']}`",
            f"- Selection metric: `{payload['config']['selection_metric']}`",
            f"- Alpha: {payload['config']['alpha']}",
            f"- min_df: {payload['config']['min_df']}",
            "",
            "## Locked test result",
            "",
            f"- Precision: {test['precision']:.4f}",
            f"- Recall: {test['recall']:.4f}",
            f"- F1: {test['f1']:.4f}",
            f"- PR-AUC: {test['pr_auc']:.4f}",
            f"- ROC-AUC: {test['roc_auc']:.4f}",
            f"- Confusion matrix: TN={test['tn']}, FP={test['fp']}, "
            f"FN={test['fn']}, TP={test['tp']}",
            "",
            "## Most spam-indicative tokens",
            "",
        ]
    )
    lines.extend(
        f"- `{token}`: log-odds {score:.4f}"
        for token, score in payload["top_spam_tokens"]
    )
    lines.extend(["", "## Most ham-indicative tokens", ""])
    lines.extend(f"- `{token}`: log-odds {score:.4f}" for token, score in payload["top_ham_tokens"])
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_m2(paths: ProjectPaths, config: M2Config | None = None) -> dict[str, Any]:
    config = config or M2Config()
    if config.selection_metric not in {"pr_auc", "f1", "precision", "recall", "roc_auc"}:
        raise ValueError(f"Unsupported selection_metric: {config.selection_metric}")
    paths.ensure_runtime_dirs()

    train_path = paths.data_processed / "train.csv"
    validation_path = paths.data_processed / "validation.csv"
    test_path = paths.data_processed / "test.csv"
    train = _load_split(train_path)
    validation = _load_split(validation_path)
    test = _load_split(test_path)

    candidates: list[tuple[ScratchMultinomialNB, TextPreprocessor, dict[str, Any]]] = []
    for variant in PREPROCESSING_VARIANTS:
        candidates.append(_evaluate_variant(train, validation, variant, config))
    candidates.sort(
        key=lambda item: (item[2][config.selection_metric], item[2]["f1"]),
        reverse=True,
    )
    selected_model, selected_preprocessor, selected_metrics = candidates[0]

    test_tokens = selected_preprocessor.transform(test["text"].astype(str))
    test_probabilities = selected_model.predict_proba(test_tokens)[:, 1].tolist()
    test_metrics = binary_metrics(test["label"].astype(int).tolist(), test_probabilities)

    model_path = paths.artifacts / "m2_scratch_nb.json"
    selected_model.save(model_path, selected_preprocessor.config)
    report_json_path = paths.reports / "m2_results.json"
    report_markdown_path = paths.reports / "m2_report.md"

    payload: dict[str, Any] = {
        "milestone": "M2",
        "model": "scratch_multinomial_naive_bayes",
        "config": asdict(config),
        "selection_protocol": "validation-only preprocessing selection; one locked test evaluation",
        "data_sha256": {
            "train": sha256_file(train_path),
            "validation": sha256_file(validation_path),
            "test": sha256_file(test_path),
        },
        "validation_results": [item[2] for item in candidates],
        "selected": selected_metrics,
        "test_metrics": test_metrics,
        "top_spam_tokens": selected_model.top_tokens(1, n=20),
        "top_ham_tokens": selected_model.top_tokens(0, n=20),
        "artifact": {
            "path": str(model_path.relative_to(paths.root)),
            "sha256": sha256_file(model_path),
            "bytes": model_path.stat().st_size,
        },
    }
    report_json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_report(payload, report_markdown_path)
    payload["report_json"] = str(report_json_path)
    payload["report_markdown"] = str(report_markdown_path)
    payload["model_path"] = str(model_path)
    return payload
