from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
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
from sklearn.pipeline import Pipeline

from spam_filter.config import ProjectPaths
from spam_filter.data import sha256_file
from spam_filter.m3 import load_m3_model
from spam_filter.m5 import load_m5_for_inference
from spam_filter.m6 import _m3_scores, _predict_transformer_batches

URL_LIKE_RE = re.compile(r"(?:https?://|www\.|\b[a-z0-9.-]+\.(?:com|net|org|co|uk)\b)", re.I)
HTML_RE = re.compile(r"<\s*(?:html|body|div|table|a|br|p)\b", re.I)
SUBJECT_RE = re.compile(r"(?im)^\s*subject\s*:")
WORD_RE = re.compile(r"[A-Za-z]{5,}")
ZERO_WIDTH = "\u200b"


@dataclass(frozen=True, slots=True)
class M7Config:
    batch_size: int = 32
    max_length: int = 128
    device: str = "auto"
    top_features: int = 25

    def validate(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.max_length < 8:
            raise ValueError("max_length must be >= 8")
        if self.device not in {"auto", "mps", "cuda", "cpu"}:
            raise ValueError("device must be one of: auto, mps, cuda, cpu")
        if self.top_features < 5:
            raise ValueError("top_features must be >= 5")


def length_bucket(length: int) -> str:
    if length <= 160:
        return "sms_like_<=160"
    if length <= 512:
        return "short_email_161_512"
    if length <= 2048:
        return "email_513_2048"
    return "long_email_>2048"


def enrich_external_frame(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    text = enriched["text"].astype(str)
    enriched["length_chars"] = text.str.len().astype(int)
    enriched["length_bucket"] = enriched["length_chars"].map(length_bucket)
    enriched["has_url_like"] = text.str.contains(URL_LIKE_RE, regex=True)
    enriched["has_digits"] = text.str.contains(r"\d", regex=True)
    enriched["has_html"] = text.str.contains(HTML_RE, regex=True)
    enriched["has_subject_header"] = text.str.contains(SUBJECT_RE, regex=True)
    return enriched


def _safe_auc(y: np.ndarray, scores: np.ndarray, metric: str) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    if metric == "pr_auc":
        return float(average_precision_score(y, scores))
    return float(roc_auc_score(y, scores))


def slice_metrics(y: np.ndarray, pred: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    negatives = tn + fp
    positives = tp + fn
    return {
        "rows": int(len(y)),
        "spam_rate": float(np.mean(y)) if len(y) else 0.0,
        "accuracy": float(accuracy_score(y, pred)) if len(y) else 0.0,
        "precision": float(precision_score(y, pred, zero_division=0)) if len(y) else 0.0,
        "recall": float(recall_score(y, pred, zero_division=0)) if len(y) else 0.0,
        "f1": float(f1_score(y, pred, zero_division=0)) if len(y) else 0.0,
        "fpr": float(fp / negatives) if negatives else 0.0,
        "fnr": float(fn / positives) if positives else 0.0,
        "pr_auc": _safe_auc(y, scores, "pr_auc"),
        "roc_auc": _safe_auc(y, scores, "roc_auc"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def external_slice_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    definitions: list[tuple[str, str, pd.Series]] = []
    for bucket in ["sms_like_<=160", "short_email_161_512", "email_513_2048", "long_email_>2048"]:
        definitions.append(("length", bucket, frame["length_bucket"].eq(bucket)))
    for column in ["has_url_like", "has_digits", "has_html", "has_subject_header"]:
        definitions.append(("feature", column, frame[column].astype(bool)))

    y_all = frame["label"].to_numpy(dtype=np.int8)
    for slice_type, slice_name, mask in definitions:
        idx = mask.to_numpy(dtype=bool)
        if not bool(idx.any()):
            continue
        y = y_all[idx]
        for model, pred_col, score_col in [
            ("m3_sparse", "m3_prediction", "m3_score"),
            ("m5_transformer", "m5_prediction", "m5_probability"),
        ]:
            metrics = slice_metrics(
                y,
                frame.loc[idx, pred_col].to_numpy(dtype=np.int8),
                frame.loc[idx, score_col].to_numpy(dtype=float),
            )
            rows.append(
                {
                    "slice_type": slice_type,
                    "slice": slice_name,
                    "model": model,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def agreement_summary(frame: pd.DataFrame) -> dict[str, int]:
    y = frame["label"].to_numpy(dtype=np.int8)
    m3_ok = frame["m3_prediction"].to_numpy(dtype=np.int8) == y
    m5_ok = frame["m5_prediction"].to_numpy(dtype=np.int8) == y
    return {
        "both_correct": int(np.sum(m3_ok & m5_ok)),
        "only_m3_correct": int(np.sum(m3_ok & ~m5_ok)),
        "only_m5_correct": int(np.sum(~m3_ok & m5_ok)),
        "both_wrong": int(np.sum(~m3_ok & ~m5_ok)),
    }


def build_error_audit(frame: pd.DataFrame) -> pd.DataFrame:
    audit = frame[
        [
            "source",
            "file_name",
            "label_name",
            "label",
            "text_sha256",
            "length_chars",
            "length_bucket",
            "has_url_like",
            "has_digits",
            "has_html",
            "has_subject_header",
            "m3_prediction",
            "m3_score",
            "m5_prediction",
            "m5_probability",
        ]
    ].copy()
    audit["m3_correct"] = audit["m3_prediction"].eq(audit["label"])
    audit["m5_correct"] = audit["m5_prediction"].eq(audit["label"])
    audit["error_group"] = np.select(
        [
            audit["m3_correct"] & audit["m5_correct"],
            audit["m3_correct"] & ~audit["m5_correct"],
            ~audit["m3_correct"] & audit["m5_correct"],
        ],
        ["both_correct", "only_m3_correct", "only_m5_correct"],
        default="both_wrong",
    )
    return audit


def _apply_to_long_words(text: str, transform: Callable[[str], str]) -> str:
    return WORD_RE.sub(lambda match: transform(match.group(0)), str(text))


def attack_leetspeak(text: str) -> str:
    table = str.maketrans({"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"})

    def mutate(word: str) -> str:
        chars = list(word)
        for index in range(1, len(chars), 3):
            lower = chars[index].lower()
            replacement = lower.translate(table)
            if replacement != lower:
                chars[index] = replacement
        return "".join(chars)

    return _apply_to_long_words(text, mutate)


def attack_separator(text: str) -> str:
    def mutate(word: str) -> str:
        midpoint = max(2, len(word) // 2)
        return word[:midpoint] + "." + word[midpoint:]

    return _apply_to_long_words(text, mutate)


def attack_zero_width(text: str) -> str:
    def mutate(word: str) -> str:
        midpoint = max(2, len(word) // 2)
        return word[:midpoint] + ZERO_WIDTH + word[midpoint:]

    return _apply_to_long_words(text, mutate)


def attack_confusable(text: str) -> str:
    confusable = {"a": "а", "c": "с", "e": "е", "o": "о", "p": "р", "x": "х", "y": "у"}

    def mutate(word: str) -> str:
        chars = list(word)
        for index in range(0, len(chars), 3):
            lower = chars[index].lower()
            if lower in confusable:
                replacement = confusable[lower]
                chars[index] = replacement.upper() if chars[index].isupper() else replacement
        return "".join(chars)

    return _apply_to_long_words(text, mutate)


ATTACKS: dict[str, Callable[[str], str]] = {
    "leetspeak": attack_leetspeak,
    "separator": attack_separator,
    "zero_width": attack_zero_width,
    "unicode_confusable": attack_confusable,
}


def _m3_predict(model: Pipeline, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(model.predict(texts), dtype=np.int8)
    scores = _m3_scores(model, texts)
    return pred, scores


def _clean_to_miss_rate(clean_pred: np.ndarray, attacked_pred: np.ndarray) -> float:
    clean_detected = clean_pred == 1
    denominator = int(np.sum(clean_detected))
    if denominator == 0:
        return 0.0
    return float(np.sum(clean_detected & (attacked_pred == 0)) / denominator)


def _robustness_rows(
    spam_texts: list[str],
    m3_model: Pipeline,
    tokenizer: Any,
    transformer: Any,
    device: str,
    config: M7Config,
) -> pd.DataFrame:
    if not spam_texts:
        raise ValueError("M7 robustness test requires at least one spam example")

    clean_m3_pred, clean_m3_score = _m3_predict(m3_model, spam_texts)
    clean_m5_prob = _predict_transformer_batches(
        spam_texts,
        tokenizer,
        transformer,
        device,
        batch_size=config.batch_size,
        max_length=config.max_length,
    )
    clean_m5_pred = (clean_m5_prob >= 0.5).astype(np.int8)

    rows: list[dict[str, Any]] = []

    def add_rows(
        attack: str,
        m3_pred: np.ndarray,
        m3_score: np.ndarray,
        m5_pred: np.ndarray,
        m5_prob: np.ndarray,
        changed_rate: float,
    ) -> None:
        rows.extend(
            [
                {
                    "attack": attack,
                    "model": "m3_sparse",
                    "spam_detection_rate": float(np.mean(m3_pred)),
                    "clean_correct_to_miss_rate": _clean_to_miss_rate(clean_m3_pred, m3_pred),
                    "attack_changed_rate": changed_rate,
                    "mean_score": float(np.mean(m3_score)),
                    "mean_score_delta_vs_clean": float(np.mean(m3_score) - np.mean(clean_m3_score)),
                },
                {
                    "attack": attack,
                    "model": "m5_transformer",
                    "spam_detection_rate": float(np.mean(m5_pred)),
                    "clean_correct_to_miss_rate": _clean_to_miss_rate(clean_m5_pred, m5_pred),
                    "attack_changed_rate": changed_rate,
                    "mean_score": float(np.mean(m5_prob)),
                    "mean_score_delta_vs_clean": float(np.mean(m5_prob) - np.mean(clean_m5_prob)),
                },
            ]
        )

    add_rows(
        "clean",
        clean_m3_pred,
        clean_m3_score,
        clean_m5_pred,
        clean_m5_prob,
        0.0,
    )
    for name, attack in ATTACKS.items():
        attacked = [attack(text) for text in spam_texts]
        m3_pred, m3_score = _m3_predict(m3_model, attacked)
        m5_prob = _predict_transformer_batches(
            attacked,
            tokenizer,
            transformer,
            device,
            batch_size=config.batch_size,
            max_length=config.max_length,
        )
        m5_pred = (m5_prob >= 0.5).astype(np.int8)
        changed_rate = float(np.mean([a != b for a, b in zip(attacked, spam_texts, strict=True)]))
        add_rows(name, m3_pred, m3_score, m5_pred, m5_prob, changed_rate)
    return pd.DataFrame(rows)


def sparse_explainability(model: Pipeline, top_k: int) -> pd.DataFrame:
    features = model.named_steps.get("features")
    classifier = model.named_steps.get("classifier")
    if features is None or classifier is None:
        raise TypeError("M3 artifact must contain features and classifier pipeline steps")
    if not hasattr(features, "get_feature_names_out") or not hasattr(classifier, "coef_"):
        raise TypeError("M3 champion does not expose linear feature coefficients")
    names = np.asarray(features.get_feature_names_out(), dtype=object)
    weights = np.asarray(classifier.coef_, dtype=float)
    if weights.ndim != 2 or weights.shape[0] != 1 or weights.shape[1] != len(names):
        raise ValueError("Unexpected M3 coefficient shape")
    coef = weights[0]
    positive = np.argsort(coef)[-top_k:][::-1]
    negative = np.argsort(coef)[:top_k]
    rows: list[dict[str, Any]] = []
    for rank, index in enumerate(positive, start=1):
        rows.append(
            {
                "direction": "spam",
                "rank": rank,
                "feature": str(names[index]),
                "weight": float(coef[index]),
            }
        )
    for rank, index in enumerate(negative, start=1):
        rows.append(
            {
                "direction": "ham",
                "rank": rank,
                "feature": str(names[index]),
                "weight": float(coef[index]),
            }
        )
    return pd.DataFrame(rows)


def _write_report(payload: dict[str, Any], path: Path) -> None:
    agreement = payload["external"]["agreement"]
    robust = pd.DataFrame(payload["robustness"]["rows"])
    lines = [
        "# M7 Robustness, Error Taxonomy & Explainability",
        "",
        "M7 is diagnostic only: it does not fit models, tune thresholds, or adapt on Enron labels.",
        "",
        "## External Enron error agreement",
        "",
        f"- Both models correct: {agreement['both_correct']}",
        f"- Only M3 sparse correct: {agreement['only_m3_correct']}",
        f"- Only M5 Transformer correct: {agreement['only_m5_correct']}",
        f"- Both wrong: {agreement['both_wrong']}",
        "",
        "## External length slices",
        "",
        "| slice | model | rows | F1 | FPR | FNR | PR-AUC |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["external"]["slice_rows"]:
        if row["slice_type"] != "length":
            continue
        pr_auc = "n/a" if row["pr_auc"] is None else f"{row['pr_auc']:.4f}"
        lines.append(
            f"| {row['slice']} | {row['model']} | {row['rows']} | {row['f1']:.4f} | "
            f"{row['fpr']:.4f} | {row['fnr']:.4f} | {pr_auc} |"
        )
    lines.extend(
        [
            "",
            "## Synthetic SMS spam evasion stress test",
            "",
            "The stress set starts from the frozen M1 test spam examples and applies "
            "deterministic, label-preserving surface perturbations. No attacked example is "
            "used for fitting.",
            "",
            "| attack | model | changed | spam detection rate | clean→miss rate | mean score | "
            "score delta |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in robust.to_dict(orient="records"):
        lines.append(
            f"| {row['attack']} | {row['model']} | {row['attack_changed_rate']:.4f} | "
            f"{row['spam_detection_rate']:.4f} | {row['clean_correct_to_miss_rate']:.4f} | "
            f"{row['mean_score']:.4f} | "
            f"{row['mean_score_delta_vs_clean']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## External slice files",
            "",
            f"- Slice metrics: `{payload['external']['slice_metrics']['path']}`",
            f"- Error audit: `{payload['external']['error_audit']['path']}`",
            f"- Sparse feature explanation: `{payload['explainability']['path']}`",
            "",
            "## Interpretation contract",
            "",
            "- Enron remains an external diagnostic benchmark; no Enron-derived adaptation "
            "occurs in M7.",
            "- Robustness attacks are synthetic stress tests, not estimates of real-world "
            "attack prevalence.",
            "- Character n-gram weights explain the linear sparse model globally; they are not "
            "causal explanations.",
            "- M8 should freeze a serving policy that clearly states its intended SMS domain and "
            "exposes domain-risk metadata.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_m7(paths: ProjectPaths, config: M7Config | None = None) -> dict[str, Any]:
    config = config or M7Config()
    config.validate()
    paths.ensure_runtime_dirs()

    external_path = paths.data_processed / "enron1_external.csv"
    predictions_path = paths.reports / "m6_external_predictions.csv"
    sms_test_path = paths.data_processed / "test.csv"
    m3_path = paths.artifacts / "m3_sparse_champion.joblib"
    m5_manifest = paths.artifacts / "m5_transformer" / "m5_manifest.json"
    required = [external_path, predictions_path, sms_test_path, m3_path, m5_manifest]
    missing = [str(path.relative_to(paths.root)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing M7 prerequisites: {missing}")

    external = pd.read_csv(external_path)
    predictions = pd.read_csv(predictions_path)
    merged = external.merge(
        predictions,
        on="text_sha256",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_pred"),
    )
    if len(merged) != len(external):
        raise ValueError(f"M6 prediction audit covers {len(merged)}/{len(external)} Enron rows")
    for canonical in ["source", "file_name", "label_name", "label"]:
        pred_col = f"{canonical}_pred"
        if pred_col in merged.columns:
            merged = merged.drop(columns=[pred_col])
    enriched = enrich_external_frame(merged)

    slices = external_slice_table(enriched)
    slices_path = paths.reports / "m7_external_slice_metrics.csv"
    slices.to_csv(slices_path, index=False)

    error_audit = build_error_audit(enriched)
    error_path = paths.reports / "m7_external_error_audit.csv"
    error_audit.to_csv(error_path, index=False)

    m3_model = load_m3_model(m3_path)
    explain = sparse_explainability(m3_model, config.top_features)
    explain_path = paths.reports / "m7_sparse_explainability.csv"
    explain.to_csv(explain_path, index=False)

    test = pd.read_csv(sms_test_path)
    spam_texts = test.loc[test["label"].astype(int).eq(1), "text"].astype(str).tolist()
    tokenizer, transformer, device = load_m5_for_inference(paths, requested_device=config.device)
    robustness = _robustness_rows(
        spam_texts,
        m3_model,
        tokenizer,
        transformer,
        device,
        config,
    )
    robustness_path = paths.reports / "m7_robustness.csv"
    robustness.to_csv(robustness_path, index=False)

    payload: dict[str, Any] = {
        "milestone": "M7",
        "config": asdict(config),
        "protocol": (
            "diagnostic only; no fitting or threshold selection on Enron or attacked samples"
        ),
        "inputs": {
            "enron_external_sha256": sha256_file(external_path),
            "m6_predictions_sha256": sha256_file(predictions_path),
            "sms_test_sha256": sha256_file(sms_test_path),
        },
        "external": {
            "rows": int(len(enriched)),
            "agreement": agreement_summary(enriched),
            "slice_rows": slices.to_dict(orient="records"),
            "slice_metrics": {
                "path": str(slices_path.relative_to(paths.root)),
                "sha256": sha256_file(slices_path),
                "rows": int(len(slices)),
            },
            "error_audit": {
                "path": str(error_path.relative_to(paths.root)),
                "sha256": sha256_file(error_path),
                "rows": int(len(error_audit)),
            },
        },
        "robustness": {
            "source": "frozen M1 test spam subset",
            "spam_rows": int(len(spam_texts)),
            "device": device,
            "rows": robustness.to_dict(orient="records"),
            "path": str(robustness_path.relative_to(paths.root)),
            "sha256": sha256_file(robustness_path),
        },
        "explainability": {
            "method": "global LinearSVC character n-gram coefficients",
            "top_spam": (
                explain.loc[explain["direction"].eq("spam")]
                .head(10)
                .to_dict(orient="records")
            ),
            "top_ham": (
                explain.loc[explain["direction"].eq("ham")]
                .head(10)
                .to_dict(orient="records")
            ),
            "path": str(explain_path.relative_to(paths.root)),
            "sha256": sha256_file(explain_path),
            "rows": int(len(explain)),
        },
    }
    results_path = paths.reports / "m7_results.json"
    report_path = paths.reports / "m7_report.md"
    results_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(payload, report_path)
    payload["report_json"] = str(results_path)
    payload["report_markdown"] = str(report_path)
    return payload


def verify_m7_outputs(paths: ProjectPaths) -> dict[str, bool]:
    return {
        "results": (paths.reports / "m7_results.json").exists(),
        "report": (paths.reports / "m7_report.md").exists(),
        "slices": (paths.reports / "m7_external_slice_metrics.csv").exists(),
        "errors": (paths.reports / "m7_external_error_audit.csv").exists(),
        "robustness": (paths.reports / "m7_robustness.csv").exists(),
        "explainability": (paths.reports / "m7_sparse_explainability.csv").exists(),
    }
