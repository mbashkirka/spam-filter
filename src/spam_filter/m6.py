from __future__ import annotations

import hashlib
import json
import shutil
import ssl
import subprocess
import tarfile
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import certifi
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from spam_filter.config import ProjectPaths
from spam_filter.data import sha256_file
from spam_filter.m3 import load_m3_model
from spam_filter.m4 import expected_calibration_error, load_m4_model
from spam_filter.m5 import load_m5_for_inference, predict_m5_probabilities

ENRON1_PRIMARY_URL = "https://www2.aueb.gr/users/ion/data/enron-spam/preprocessed/enron1.tar.gz"
ENRON1_FALLBACK_URL = "https://www.aueb.gr/users/ion/data/enron-spam/preprocessed/enron1.tar.gz"
ENRON1_EXPECTED_HAM = 3672
ENRON1_EXPECTED_SPAM = 1500


@dataclass(frozen=True, slots=True)
class M6Config:
    dataset: str = "enron1"
    batch_size: int = 32
    max_length: int = 128
    device: str = "auto"
    calibration_bins: int = 10
    strict_counts: bool = True

    def validate(self) -> None:
        if self.dataset != "enron1":
            raise ValueError("M6 currently supports dataset='enron1' only")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.max_length < 8:
            raise ValueError("max_length must be >= 8")
        if self.device not in {"auto", "mps", "cuda", "cpu"}:
            raise ValueError("device must be one of: auto, mps, cuda, cpu")
        if self.calibration_bins < 2:
            raise ValueError("calibration_bins must be >= 2")


@dataclass(frozen=True, slots=True)
class PreparedExternalDataset:
    csv_path: Path
    manifest_path: Path
    rows_raw: int
    rows_unique: int
    duplicates_removed: int


def _verified_ssl_context() -> ssl.SSLContext:
    """Return a TLS context backed by certifi instead of the host Python CA bundle."""
    return ssl.create_default_context(cafile=certifi.where())


def _download_bytes(url: str, timeout_seconds: int = 60) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "spam-filter-lab/0.7.1 (+portfolio research project)"},
    )
    context = _verified_ssl_context()
    with urllib.request.urlopen(  # noqa: S310
        request, timeout=timeout_seconds, context=context
    ) as response:
        return response.read()


def _download_with_curl(url: str, destination: Path, timeout_seconds: int = 180) -> None:
    """Use the system curl as a verified-TLS fallback (never disables certificate checks)."""
    curl = shutil.which("curl")
    if curl is None:
        raise RuntimeError("system curl is not available")
    subprocess.run(  # noqa: S603
        [
            curl,
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--retry",
            "3",
            "--connect-timeout",
            "15",
            "--max-time",
            str(timeout_seconds),
            "--output",
            str(destination),
            url,
        ],
        check=True,
    )


def _validate_downloaded_archive(path: Path) -> None:
    if not path.exists():
        raise RuntimeError("download did not create an archive")
    if path.stat().st_size < 10_000:
        raise RuntimeError(f"downloaded payload is unexpectedly small: {path.stat().st_size} bytes")
    try:
        with tarfile.open(path, mode="r:gz") as archive:
            names = [member.name.replace("\\", "/") for member in archive.getmembers()]
    except (tarfile.TarError, OSError) as exc:
        raise RuntimeError(f"downloaded file is not a valid gzip tar archive: {exc}") from exc
    has_ham = any("/ham/" in f"/{name}/" for name in names)
    has_spam = any("/spam/" in f"/{name}/" for name in names)
    if not (has_ham and has_spam):
        raise RuntimeError("downloaded archive does not contain both ham and spam directories")


def download_enron1_archive(destination: Path) -> str:
    """Download official AUEB Enron1 using certifi TLS, then system curl as fallback."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    urls = (ENRON1_PRIMARY_URL, ENRON1_FALLBACK_URL)

    for url in urls:
        temp_path = destination.with_suffix(destination.suffix + ".part")
        temp_path.unlink(missing_ok=True)
        try:
            payload = _download_bytes(url)
            temp_path.write_bytes(payload)
            _validate_downloaded_archive(temp_path)
            temp_path.replace(destination)
            return url
        except (OSError, RuntimeError) as exc:
            temp_path.unlink(missing_ok=True)
            errors.append(f"urllib+certifi {url}: {exc}")

    for url in urls:
        temp_path = destination.with_suffix(destination.suffix + ".part")
        temp_path.unlink(missing_ok=True)
        try:
            _download_with_curl(url, temp_path)
            _validate_downloaded_archive(temp_path)
            temp_path.replace(destination)
            return url
        except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
            temp_path.unlink(missing_ok=True)
            errors.append(f"curl {url}: {exc}")

    raise RuntimeError(
        "Unable to download Enron1 from AUEB with verified TLS. "
        + " | ".join(errors)
        + ". You can also download enron1.tar.gz manually into data/raw/enron1.tar.gz "
        + "and rerun `make m6-data`."
    )


def _decode_message(payload: bytes) -> str:
    try:
        return payload.decode("utf-8").strip()
    except UnicodeDecodeError:
        return payload.decode("latin-1", errors="replace").strip()


def parse_enron1_archive(archive_path: Path) -> pd.DataFrame:
    if not archive_path.exists():
        raise FileNotFoundError(f"Enron1 archive not found: {archive_path}")

    rows: list[dict[str, Any]] = []
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            normalized = member.name.replace("\\", "/")
            parts = [part for part in normalized.split("/") if part]
            if len(parts) < 3:
                continue
            if "ham" in parts:
                label_name, label = "ham", 0
            elif "spam" in parts:
                label_name, label = "spam", 1
            else:
                continue
            file_obj = archive.extractfile(member)
            if file_obj is None:
                continue
            text = _decode_message(file_obj.read())
            if not text:
                continue
            text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
            rows.append(
                {
                    "source": "enron1",
                    "file_name": Path(normalized).name,
                    "text": text,
                    "label_name": label_name,
                    "label": label,
                    "text_sha256": text_sha256,
                }
            )

    if not rows:
        raise ValueError("No ham/spam messages were found in the Enron1 archive")
    frame = pd.DataFrame(rows)
    conflicts = frame.groupby("text_sha256")["label"].nunique()
    if bool((conflicts > 1).any()):
        count = int((conflicts > 1).sum())
        raise ValueError(f"Enron1 contains {count} exact-text label conflicts")
    return frame.sort_values(["label", "file_name"], kind="stable").reset_index(drop=True)


def prepare_enron1_dataset(
    paths: ProjectPaths, *, strict_counts: bool = True
) -> PreparedExternalDataset:
    paths.ensure_runtime_dirs()
    archive_path = paths.data_raw / "enron1.tar.gz"
    source_url = ENRON1_PRIMARY_URL
    if not archive_path.exists():
        source_url = download_enron1_archive(archive_path)

    raw = parse_enron1_archive(archive_path)
    ham_raw = int((raw["label"] == 0).sum())
    spam_raw = int((raw["label"] == 1).sum())
    if strict_counts and (ham_raw != ENRON1_EXPECTED_HAM or spam_raw != ENRON1_EXPECTED_SPAM):
        raise ValueError(
            "Unexpected Enron1 class counts: "
            f"ham={ham_raw}, spam={spam_raw}; expected "
            f"ham={ENRON1_EXPECTED_HAM}, spam={ENRON1_EXPECTED_SPAM}"
        )

    unique = raw.drop_duplicates(subset=["text_sha256"], keep="first").reset_index(drop=True)
    csv_path = paths.data_processed / "enron1_external.csv"
    unique.to_csv(csv_path, index=False)

    manifest = {
        "milestone": "M6",
        "dataset": "Enron-Spam / Enron1",
        "source": "Athens University of Economics and Business (AUEB)",
        "source_url": source_url,
        "primary_url": ENRON1_PRIMARY_URL,
        "archive": {
            "path": str(archive_path.relative_to(paths.root)),
            "sha256": sha256_file(archive_path),
            "bytes": archive_path.stat().st_size,
        },
        "processed": {
            "path": str(csv_path.relative_to(paths.root)),
            "sha256": sha256_file(csv_path),
            "rows_raw": int(len(raw)),
            "rows_unique": int(len(unique)),
            "duplicates_removed": int(len(raw) - len(unique)),
            "ham_raw": ham_raw,
            "spam_raw": spam_raw,
            "ham_unique": int((unique["label"] == 0).sum()),
            "spam_unique": int((unique["label"] == 1).sum()),
            "spam_rate_unique": float(unique["label"].mean()),
            "median_chars": float(unique["text"].str.len().median()),
            "p95_chars": float(unique["text"].str.len().quantile(0.95)),
        },
        "protocol": (
            "external zero-shot benchmark; no Enron labels are used for fitting or selection"
        ),
    }
    manifest_path = paths.data_processed / "m6_enron1_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return PreparedExternalDataset(
        csv_path=csv_path,
        manifest_path=manifest_path,
        rows_raw=int(len(raw)),
        rows_unique=int(len(unique)),
        duplicates_removed=int(len(raw) - len(unique)),
    )


def _binary_metrics(
    y_true: np.ndarray, predictions: np.ndarray, scores: np.ndarray
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


def _probability_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, bins: int
) -> dict[str, float]:
    prob = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1 - 1e-7)
    return {
        "brier": float(brier_score_loss(y_true, prob)),
        "log_loss": float(log_loss(y_true, prob, labels=[0, 1])),
        "ece": float(expected_calibration_error(y_true, prob, bins=bins)),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
    }


def _m3_scores(model: Any, texts: list[str]) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(texts), dtype=float)
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(texts), dtype=float)[:, 1]
    raise TypeError("M3 model must implement decision_function or predict_proba")


def _predict_transformer_batches(
    texts: list[str], tokenizer: Any, model: Any, device: str, *, batch_size: int, max_length: int
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        chunks.append(
            predict_m5_probabilities(
                texts[start : start + batch_size],
                tokenizer,
                model,
                device,
                max_length=max_length,
            )
        )
    return np.concatenate(chunks) if chunks else np.asarray([], dtype=float)


def _load_internal_benchmarks(paths: ProjectPaths) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for milestone in ("m3", "m4", "m5"):
        path = paths.reports / f"{milestone}_results.json"
        if path.exists():
            payload[milestone] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def _write_report(payload: dict[str, Any], path: Path) -> None:
    data = payload["dataset"]
    m3 = payload["models"]["m3_sparse"]
    m4 = payload["models"]["m4_calibrated_sparse"]
    m5 = payload["models"]["m5_transformer"]
    lines = [
        "# M6 External Domain-Shift Report: SMS → Enron Email",
        "",
        "M6 does not fit, calibrate, select, or tune any model on Enron labels. The previously "
        "frozen SMS models are evaluated zero-shot on the external Enron1 email corpus.",
        "",
        "## External dataset",
        "",
        "- Dataset: Enron-Spam / Enron1 (AUEB)",
        f"- Raw messages: {data['rows_raw']}",
        f"- Unique exact texts evaluated: {data['rows_unique']}",
        f"- Exact duplicates removed for evaluation: {data['duplicates_removed']}",
        f"- Ham/spam (unique): {data['ham_unique']}/{data['spam_unique']}",
        f"- Spam rate: {data['spam_rate_unique']:.4f}",
        f"- Median length: {data['median_chars']:.1f} chars",
        f"- 95th percentile length: {data['p95_chars']:.1f} chars",
        "",
        "## Zero-shot external benchmark",
        "",
        "| frozen model/policy | precision | recall | F1 | PR-AUC | ROC-AUC | FP | FN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| M3 char-TF-IDF + LinearSVC | {m3['precision']:.4f} | {m3['recall']:.4f} | "
        f"{m3['f1']:.4f} | {m3['pr_auc']:.4f} | {m3['roc_auc']:.4f} | {m3['fp']} | {m3['fn']} |",
        f"| M4 calibrated sparse @ {m4['threshold']:.6f} | {m4['precision']:.4f} | "
        f"{m4['recall']:.4f} | {m4['f1']:.4f} | {m4['pr_auc']:.4f} | {m4['roc_auc']:.4f} | "
        f"{m4['fp']} | {m4['fn']} |",
        f"| M5 DistilBERT @ 0.5 | {m5['precision']:.4f} | {m5['recall']:.4f} | "
        f"{m5['f1']:.4f} | {m5['pr_auc']:.4f} | {m5['roc_auc']:.4f} | {m5['fp']} | {m5['fn']} |",
        "",
        "## Probability quality under domain shift",
        "",
        "| probability model | Brier ↓ | log loss ↓ | ECE ↓ |",
        "|---|---:|---:|---:|",
        f"| M4 calibrated sparse | {m4['probability']['brier']:.4f} | "
        f"{m4['probability']['log_loss']:.4f} | {m4['probability']['ece']:.4f} |",
        f"| M5 DistilBERT | {m5['probability']['brier']:.4f} | "
        f"{m5['probability']['log_loss']:.4f} | {m5['probability']['ece']:.4f} |",
        "",
        "## Methodological interpretation",
        "",
        "- Enron1 is an external domain-shift benchmark, not an additional training split.",
        (
            "- No threshold is re-selected on Enron1. M4 keeps its SMS-validation threshold "
            "and M5 keeps 0.5."
        ),
        "- Any drop from the SMS benchmark quantifies transfer risk rather than training failure.",
        (
            "- The external email benchmark is substantially longer and stylistically "
            "different from SMS."
        ),
        (
            "- M7 should analyze errors and adversarial/obfuscated slices before a serving "
            "model is frozen."
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_m6(paths: ProjectPaths, config: M6Config | None = None) -> dict[str, Any]:
    config = config or M6Config()
    config.validate()
    paths.ensure_runtime_dirs()

    csv_path = paths.data_processed / "enron1_external.csv"
    manifest_path = paths.data_processed / "m6_enron1_manifest.json"
    if not csv_path.exists() or not manifest_path.exists():
        prepare_enron1_dataset(paths, strict_counts=config.strict_counts)

    frame = pd.read_csv(csv_path)
    required = {"text", "label", "text_sha256"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"M6 external dataset is missing columns: {sorted(missing)}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    texts = frame["text"].astype(str).tolist()
    labels = frame["label"].astype(int).to_numpy(dtype=np.int8)

    m3_path = paths.artifacts / "m3_sparse_champion.joblib"
    if not m3_path.exists():
        raise FileNotFoundError("M3 sparse artifact not found. Run `make m3-train` first.")
    m3_model = load_m3_model(m3_path)
    m3_predictions = np.asarray(m3_model.predict(texts), dtype=np.int8)
    m3_scores = _m3_scores(m3_model, texts)
    m3_metrics = _binary_metrics(labels, m3_predictions, m3_scores)

    m4_path = paths.artifacts / "m4_calibrated_champion.joblib"
    if not m4_path.exists():
        raise FileNotFoundError("M4 calibrated artifact not found. Run `make m4-train` first.")
    m4_model = load_m4_model(m4_path)
    m4_prob = np.asarray(m4_model.predict_proba(texts), dtype=float)[:, 1]
    m4_predictions = np.asarray(m4_model.predict(texts), dtype=np.int8)
    m4_metrics = _binary_metrics(labels, m4_predictions, m4_prob)
    m4_metrics["threshold"] = float(m4_model.threshold)
    m4_metrics["fp_cost"] = float(m4_model.fp_cost)
    m4_metrics["fn_cost"] = float(m4_model.fn_cost)
    m4_metrics["weighted_cost"] = float(
        m4_metrics["fp"] * m4_model.fp_cost + m4_metrics["fn"] * m4_model.fn_cost
    )
    m4_metrics["probability"] = _probability_metrics(labels, m4_prob, config.calibration_bins)

    tokenizer, transformer, device = load_m5_for_inference(paths, requested_device=config.device)
    m5_prob = _predict_transformer_batches(
        texts,
        tokenizer,
        transformer,
        device,
        batch_size=config.batch_size,
        max_length=config.max_length,
    )
    m5_predictions = (m5_prob >= 0.5).astype(np.int8)
    m5_metrics = _binary_metrics(labels, m5_predictions, m5_prob)
    m5_metrics["threshold"] = 0.5
    m5_metrics["device"] = device
    m5_metrics["probability"] = _probability_metrics(labels, m5_prob, config.calibration_bins)

    predictions = frame[["source", "file_name", "label_name", "label", "text_sha256"]].copy()
    predictions["m3_prediction"] = m3_predictions
    predictions["m3_score"] = m3_scores
    predictions["m4_probability"] = m4_prob
    predictions["m4_prediction"] = m4_predictions
    predictions["m5_probability"] = m5_prob
    predictions["m5_prediction"] = m5_predictions
    predictions_path = paths.reports / "m6_external_predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    processed = manifest["processed"]
    payload: dict[str, Any] = {
        "milestone": "M6",
        "config": asdict(config),
        "protocol": (
            "zero-shot external evaluation only; no Enron labels used for model fitting, "
            "calibration, model selection, or threshold selection"
        ),
        "dataset": processed,
        "dataset_manifest": str(manifest_path.relative_to(paths.root)),
        "dataset_sha256": sha256_file(csv_path),
        "models": {
            "m3_sparse": m3_metrics,
            "m4_calibrated_sparse": m4_metrics,
            "m5_transformer": m5_metrics,
        },
        "internal_sms_benchmarks": _load_internal_benchmarks(paths),
        "predictions": {
            "path": str(predictions_path.relative_to(paths.root)),
            "sha256": sha256_file(predictions_path),
            "rows": int(len(predictions)),
        },
    }
    results_path = paths.reports / "m6_results.json"
    report_path = paths.reports / "m6_report.md"
    results_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(payload, report_path)
    payload["report_json"] = str(results_path)
    payload["report_markdown"] = str(report_path)
    return payload


def verify_m6_inputs(paths: ProjectPaths) -> dict[str, bool]:
    return {
        "dataset": (paths.data_processed / "enron1_external.csv").exists(),
        "manifest": (paths.data_processed / "m6_enron1_manifest.json").exists(),
        "m3": (paths.artifacts / "m3_sparse_champion.joblib").exists(),
        "m4": (paths.artifacts / "m4_calibrated_champion.joblib").exists(),
        "m5": (paths.artifacts / "m5_transformer" / "m5_manifest.json").exists(),
    }
