from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from spam_filter.config import ProjectPaths
from spam_filter.data import sha256_file
from spam_filter.m3 import M3Config, build_candidates


@dataclass(frozen=True, slots=True)
class M4Config:
    calibration_folds: int = 5
    random_state: int = 42
    fp_cost: float = 10.0
    fn_cost: float = 1.0
    calibration_bins: int = 10

    def validate(self) -> None:
        if self.calibration_folds < 2:
            raise ValueError("calibration_folds must be >= 2")
        if self.fp_cost < 0 or self.fn_cost < 0:
            raise ValueError("misclassification costs must be non-negative")
        if self.fp_cost == 0 and self.fn_cost == 0:
            raise ValueError("at least one misclassification cost must be positive")
        if self.calibration_bins < 2:
            raise ValueError("calibration_bins must be >= 2")


def _model_calibration_scores(model: Pipeline, texts: list[str]) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(texts), dtype=float)
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(texts), dtype=float)[:, 1]
        clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
        return np.log(clipped / (1 - clipped))
    raise TypeError("M3 champion must implement decision_function or predict_proba")


@dataclass(slots=True)
class CalibratedSpamModel:
    """Frozen M3 classifier plus Platt calibration and a product decision threshold."""

    base_model: Pipeline
    calibrator: LogisticRegression
    threshold: float
    fp_cost: float
    fn_cost: float
    selected_model_name: str

    def decision_function(self, texts: list[str]) -> np.ndarray:
        return _model_calibration_scores(self.base_model, texts)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        margins = self.decision_function(texts).reshape(-1, 1)
        return np.asarray(self.calibrator.predict_proba(margins), dtype=float)

    def predict(self, texts: list[str]) -> np.ndarray:
        spam_prob = self.predict_proba(texts)[:, 1]
        return (spam_prob >= self.threshold).astype(np.int8)


def expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    y = np.asarray(y_true, dtype=float)
    prob = np.asarray(probabilities, dtype=float)
    if y.shape != prob.shape:
        raise ValueError("y_true and probabilities must have the same shape")
    if len(y) == 0:
        raise ValueError("calibration input cannot be empty")
    if np.any((prob < 0) | (prob > 1)):
        raise ValueError("probabilities must be in [0, 1]")
    if bins < 2:
        raise ValueError("bins must be >= 2")

    edges = np.linspace(0.0, 1.0, bins + 1)
    # Keep probability 1.0 in the final bin.
    assignments = np.minimum(np.digitize(prob, edges[1:-1], right=False), bins - 1)
    ece = 0.0
    for idx in range(bins):
        mask = assignments == idx
        if not np.any(mask):
            continue
        accuracy = float(np.mean(y[mask]))
        confidence = float(np.mean(prob[mask]))
        ece += float(np.mean(mask)) * abs(accuracy - confidence)
    return float(ece)


def probability_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, bins: int
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=np.int8)
    prob = np.asarray(probabilities, dtype=float)
    clipped = np.clip(prob, 1e-7, 1 - 1e-7)
    return {
        "brier": float(brier_score_loss(y, prob)),
        "log_loss": float(log_loss(y, clipped, labels=[0, 1])),
        "ece": expected_calibration_error(y, prob, bins=bins),
        "pr_auc": float(average_precision_score(y, prob)),
        "roc_auc": float(roc_auc_score(y, prob)),
    }


def _decision_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    fp_cost: float,
    fn_cost: float,
) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=np.int8)
    pred = (np.asarray(probabilities, dtype=float) >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    total_cost = float(fp * fp_cost + fn * fn_cost)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "total_cost": total_cost,
        "cost_per_message": total_cost / len(y),
    }


def threshold_curve(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    fp_cost: float,
    fn_cost: float,
) -> list[dict[str, Any]]:
    prob = np.asarray(probabilities, dtype=float)
    if len(prob) == 0:
        raise ValueError("probabilities cannot be empty")
    unique = np.unique(prob)
    midpoints = (unique[:-1] + unique[1:]) / 2 if len(unique) > 1 else np.array([])
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], midpoints)))
    return [
        _decision_metrics(y_true, prob, float(threshold), fp_cost, fn_cost)
        for threshold in candidates
    ]


def select_cost_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    fp_cost: float,
    fn_cost: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    curve = threshold_curve(y_true, probabilities, fp_cost, fn_cost)
    # Product preference under equal cost: fewer false positives, then better recall/F1.
    selected = min(
        curve,
        key=lambda row: (
            row["total_cost"],
            row["fp"],
            row["fn"],
            -row["f1"],
            abs(row["threshold"] - 0.5),
        ),
    )
    return selected, curve


def _load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required M1 split not found: {path}. Run `make m1-data` first.")
    frame = pd.read_csv(path)
    missing = {"text", "label"}.difference(frame.columns)
    if missing:
        raise ValueError(f"Split {path} is missing columns: {sorted(missing)}")
    return frame


def _load_m3_selection(paths: ProjectPaths) -> tuple[str, dict[str, Any]]:
    report = paths.reports / "m3_results.json"
    artifact = paths.artifacts / "m3_sparse_champion.joblib"
    if not report.exists() or not artifact.exists():
        raise FileNotFoundError("M3 champion is missing. Run `make m3-train` before M4.")
    payload = json.loads(report.read_text(encoding="utf-8"))
    selected_name = str(payload["selected_model"])
    if selected_name not in build_candidates(M3Config()):
        raise ValueError(f"Unknown M3 selected model: {selected_name}")
    return selected_name, payload


def fit_oof_platt_calibrator(
    texts: list[str],
    labels: np.ndarray,
    selected_model_name: str,
    config: M4Config,
) -> tuple[Pipeline, LogisticRegression, np.ndarray]:
    labels_array = np.asarray(labels, dtype=np.int8)
    candidates = build_candidates(M3Config(random_state=config.random_state))
    if selected_model_name not in candidates:
        raise ValueError(f"Unknown M3 model: {selected_model_name}")
    template = candidates[selected_model_name]

    folds = StratifiedKFold(
        n_splits=config.calibration_folds,
        shuffle=True,
        random_state=config.random_state,
    )
    oof_margins = np.empty(len(labels_array), dtype=float)
    indices = np.arange(len(labels_array))
    text_array = np.asarray(texts, dtype=object)
    for fit_idx, score_idx in folds.split(indices, labels_array):
        fold_model = clone(template)
        fold_model.fit(text_array[fit_idx].tolist(), labels_array[fit_idx])
        oof_margins[score_idx] = _model_calibration_scores(
            fold_model, text_array[score_idx].tolist()
        )

    # High C keeps Platt scaling close to a pure two-parameter sigmoid mapping.
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2_000)
    calibrator.fit(oof_margins.reshape(-1, 1), labels_array)

    final_model = clone(template)
    final_model.fit(texts, labels_array)
    return final_model, calibrator, oof_margins


def _calibrated_probabilities(
    model: Pipeline, calibrator: LogisticRegression, texts: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    margins = _model_calibration_scores(model, texts)
    probabilities = np.asarray(calibrator.predict_proba(margins.reshape(-1, 1)), dtype=float)[:, 1]
    return margins, probabilities


def _write_report(payload: dict[str, Any], path: Path) -> None:
    cfg = payload["config"]
    validation_prob = payload["validation_probability_metrics"]
    naive_prob = payload["validation_naive_sigmoid_metrics"]
    default = payload["validation_default_threshold"]
    selected = payload["selected_threshold"]
    test = payload["test_decision_metrics"]
    test_prob = payload["test_probability_metrics"]

    lines = [
        "# M4 Calibration & Cost-Sensitive Decision Report",
        "",
        "M4 keeps the M3 classifier family frozen. Platt calibration is fit from out-of-fold "
        "M1-train decision margins, while the product threshold is selected on the M1 validation "
        "split. The frozen test split is used only to benchmark the selected M4 decision policy.",
        "",
        "## Decision policy",
        "",
        f"- M3 champion: `{payload['selected_model']}`",
        f"- Calibration: out-of-fold Platt scaling ({cfg['calibration_folds']} folds)",
        f"- False-positive cost: {cfg['fp_cost']:.2f}",
        f"- False-negative cost: {cfg['fn_cost']:.2f}",
        (
            "- FP:FN cost ratio: "
            f"{cfg['fp_cost'] / cfg['fn_cost'] if cfg['fn_cost'] else float('inf'):.2f}:1"
        ),
        (
            "- Important: the cost ratio is a configurable product assumption, "
            "not a property of the dataset."
        ),
        "",
        "## Probability calibration on validation",
        "",
        "| probability mapping | Brier ↓ | log loss ↓ | ECE ↓ | PR-AUC ↑ | ROC-AUC ↑ |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| naive sigmoid(raw model score) | {naive_prob['brier']:.4f} | "
            f"{naive_prob['log_loss']:.4f} | {naive_prob['ece']:.4f} | "
            f"{naive_prob['pr_auc']:.4f} | {naive_prob['roc_auc']:.4f} |"
        ),
        (
            f"| OOF Platt calibrated | {validation_prob['brier']:.4f} | "
            f"{validation_prob['log_loss']:.4f} | {validation_prob['ece']:.4f} | "
            f"{validation_prob['pr_auc']:.4f} | {validation_prob['roc_auc']:.4f} |"
        ),
        "",
        "PR-AUC/ROC-AUC should remain almost unchanged under a monotonic calibration mapping; "
        "Brier score, log loss and ECE measure probability quality.",
        "",
        "## Validation operating point",
        "",
        "| policy | threshold | precision | recall | F1 | FP | FN | weighted cost |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| probability 0.5 | {default['threshold']:.6f} | {default['precision']:.4f} | "
        f"{default['recall']:.4f} | {default['f1']:.4f} | {default['fp']} | {default['fn']} | "
        f"{default['total_cost']:.2f} |",
        f"| selected cost threshold | {selected['threshold']:.6f} | {selected['precision']:.4f} | "
        f"{selected['recall']:.4f} | {selected['f1']:.4f} | {selected['fp']} | {selected['fn']} | "
        f"{selected['total_cost']:.2f} |",
        "",
        "## Frozen test benchmark",
        "",
        f"- Selected threshold: {selected['threshold']:.6f}",
        f"- Precision: {test['precision']:.4f}",
        f"- Recall: {test['recall']:.4f}",
        f"- F1: {test['f1']:.4f}",
        f"- Confusion matrix: TN={test['tn']}, FP={test['fp']}, FN={test['fn']}, TP={test['tp']}",
        f"- Weighted cost: {test['total_cost']:.2f} ({test['cost_per_message']:.6f} per message)",
        f"- Test Brier score: {test_prob['brier']:.4f}",
        f"- Test log loss: {test_prob['log_loss']:.4f}",
        f"- Test ECE: {test_prob['ece']:.4f}",
        f"- Test PR-AUC: {test_prob['pr_auc']:.4f}",
        f"- Test ROC-AUC: {test_prob['roc_auc']:.4f}",
        "",
        "## Methodological notes",
        "",
        "- The base M3 family is not re-selected in M4.",
        (
            "- Calibration does not use the validation or test labels: its sigmoid is "
            "learned from OOF train scores."
        ),
        (
            "- The operating threshold is selected only on validation using the declared "
            "FP/FN cost function."
        ),
        (
            "- The test split has already been reported in earlier milestones, so it is a "
            "frozen benchmark, not a pristine final lockbox."
        ),
        (
            "- M5/M6 should introduce additional external/domain-shift evidence before "
            "making production claims."
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_m4(paths: ProjectPaths, config: M4Config | None = None) -> dict[str, Any]:
    config = config or M4Config()
    config.validate()
    paths.ensure_runtime_dirs()
    selected_name, m3_payload = _load_m3_selection(paths)

    train_path = paths.data_processed / "train.csv"
    validation_path = paths.data_processed / "validation.csv"
    test_path = paths.data_processed / "test.csv"
    train = _load_split(train_path)
    validation = _load_split(validation_path)
    test = _load_split(test_path)

    train_texts = train["text"].astype(str).tolist()
    train_labels = train["label"].astype(int).to_numpy(dtype=np.int8)
    validation_texts = validation["text"].astype(str).tolist()
    validation_labels = validation["label"].astype(int).to_numpy(dtype=np.int8)
    test_texts = test["text"].astype(str).tolist()
    test_labels = test["label"].astype(int).to_numpy(dtype=np.int8)

    model, calibrator, oof_margins = fit_oof_platt_calibrator(
        train_texts, train_labels, selected_name, config
    )
    validation_margins, validation_prob = _calibrated_probabilities(
        model, calibrator, validation_texts
    )
    _, test_prob = _calibrated_probabilities(model, calibrator, test_texts)

    validation_prob_metrics = probability_metrics(
        validation_labels, validation_prob, bins=config.calibration_bins
    )
    validation_naive_metrics = probability_metrics(
        validation_labels, expit(validation_margins), bins=config.calibration_bins
    )
    test_prob_metrics = probability_metrics(test_labels, test_prob, bins=config.calibration_bins)

    selected_threshold, curve = select_cost_threshold(
        validation_labels, validation_prob, config.fp_cost, config.fn_cost
    )
    default_threshold = _decision_metrics(
        validation_labels, validation_prob, 0.5, config.fp_cost, config.fn_cost
    )
    test_decision = _decision_metrics(
        test_labels,
        test_prob,
        float(selected_threshold["threshold"]),
        config.fp_cost,
        config.fn_cost,
    )

    artifact = CalibratedSpamModel(
        base_model=model,
        calibrator=calibrator,
        threshold=float(selected_threshold["threshold"]),
        fp_cost=config.fp_cost,
        fn_cost=config.fn_cost,
        selected_model_name=selected_name,
    )
    artifact_path = paths.artifacts / "m4_calibrated_champion.joblib"
    joblib.dump(artifact, artifact_path, compress=3)

    curve_path = paths.reports / "m4_threshold_curve.csv"
    pd.DataFrame(curve).sort_values("threshold").to_csv(curve_path, index=False)

    payload: dict[str, Any] = {
        "milestone": "M4",
        "config": asdict(config),
        "selected_model": selected_name,
        "m3_test_metrics": m3_payload.get("test_metrics"),
        "calibration_protocol": (
            "5-fold OOF train scores -> Platt sigmoid; final base model fit on all train"
        ),
        "threshold_protocol": (
            "minimize FP/FN weighted count on validation; evaluate once as M4 policy "
            "on frozen test"
        ),
        "data_sha256": {
            "train": sha256_file(train_path),
            "validation": sha256_file(validation_path),
            "test": sha256_file(test_path),
        },
        "oof_train": {
            "rows": int(len(oof_margins)),
            "margin_mean": float(np.mean(oof_margins)),
            "margin_std": float(np.std(oof_margins)),
        },
        "validation_naive_sigmoid_metrics": validation_naive_metrics,
        "validation_probability_metrics": validation_prob_metrics,
        "validation_default_threshold": default_threshold,
        "selected_threshold": selected_threshold,
        "test_probability_metrics": test_prob_metrics,
        "test_decision_metrics": test_decision,
        "artifact": {
            "path": str(artifact_path.relative_to(paths.root)),
            "sha256": sha256_file(artifact_path),
            "bytes": artifact_path.stat().st_size,
        },
        "threshold_curve": {
            "path": str(curve_path.relative_to(paths.root)),
            "rows": len(curve),
        },
    }

    json_path = paths.reports / "m4_results.json"
    md_path = paths.reports / "m4_report.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(payload, md_path)
    payload["artifact_path"] = str(artifact_path)
    payload["report_json"] = str(json_path)
    payload["report_markdown"] = str(md_path)
    payload["threshold_curve_path"] = str(curve_path)
    return payload


def load_m4_model(path: Path) -> CalibratedSpamModel:
    model = joblib.load(path)
    if not isinstance(model, CalibratedSpamModel):
        raise TypeError("M4 artifact is not a CalibratedSpamModel")
    return model
