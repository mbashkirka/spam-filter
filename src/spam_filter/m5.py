from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from spam_filter.config import ProjectPaths
from spam_filter.data import sha256_file
from spam_filter.m4 import expected_calibration_error, probability_metrics


@dataclass(frozen=True, slots=True)
class M5Config:
    model_name: str = "distilbert-base-uncased"
    epochs: int = 3
    batch_size: int = 16
    max_length: int = 128
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    patience: int = 1
    random_state: int = 42
    device: str = "auto"
    calibration_bins: int = 10

    def validate(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name cannot be empty")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.max_length < 8:
            raise ValueError("max_length must be >= 8")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        if self.patience < 0:
            raise ValueError("patience must be >= 0")
        if self.device not in {"auto", "mps", "cuda", "cpu"}:
            raise ValueError("device must be one of: auto, mps, cuda, cpu")
        if self.calibration_bins < 2:
            raise ValueError("calibration_bins must be >= 2")


def _require_transformer_stack() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised by user environment contract
        raise RuntimeError(
            "M5 requires the transformer extra. Install with "
            '`python -m pip install -e ".[dev,data,transformer]"`.'
        ) from exc
    return torch, DataLoader, Dataset, (AutoTokenizer, AutoModelForSequenceClassification)


def resolve_device(requested: str = "auto") -> str:
    if requested not in {"auto", "mps", "cuda", "cpu"}:
        raise ValueError("device must be one of: auto, mps, cuda, cpu")
    torch, _, _, _ = _require_transformer_stack()
    if requested == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is not available in this PyTorch installation")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return requested


def compute_class_weights(labels: np.ndarray | list[int]) -> np.ndarray:
    y = np.asarray(labels, dtype=np.int8)
    counts = np.bincount(y, minlength=2).astype(float)
    if len(y) == 0 or np.any(counts == 0):
        raise ValueError("class weights require both classes to be present")
    return len(y) / (2.0 * counts)


def transformer_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=np.int8)
    prob = np.asarray(probabilities, dtype=float)
    pred = (prob >= 0.5).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y, prob)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def select_best_epoch(history: list[dict[str, Any]]) -> dict[str, Any]:
    if not history:
        raise ValueError("training history cannot be empty")
    return max(history, key=lambda row: (float(row["validation_pr_auc"]), -int(row["epoch"])))


def _load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required M1 split not found: {path}. Run `make m1-data` first.")
    frame = pd.read_csv(path)
    missing = {"text", "label"}.difference(frame.columns)
    if missing:
        raise ValueError(f"Split {path} is missing columns: {sorted(missing)}")
    return frame


def _seed_everything(seed: int, torch: Any) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_dataset_class(base_dataset: Any) -> type[Any]:
    class TextDataset(base_dataset):
        def __init__(self, texts: list[str], labels: list[int]) -> None:
            self.texts = texts
            self.labels = labels

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, index: int) -> tuple[str, int]:
            return self.texts[index], self.labels[index]

    return TextDataset


def _make_collate_fn(tokenizer: Any, max_length: int, torch: Any) -> Any:
    def collate(batch: list[tuple[str, int]]) -> dict[str, Any]:
        texts, labels = zip(*batch, strict=True)
        encoded = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(labels, dtype=torch.long)
        return encoded

    return collate


def _make_loader(
    frame: pd.DataFrame,
    tokenizer: Any,
    config: M5Config,
    torch: Any,
    data_loader_cls: Any,
    dataset_base: Any,
    *,
    shuffle: bool,
) -> Any:
    dataset_cls = _make_dataset_class(dataset_base)
    dataset = dataset_cls(
        frame["text"].astype(str).tolist(),
        frame["label"].astype(int).tolist(),
    )
    generator = torch.Generator()
    generator.manual_seed(config.random_state)
    return data_loader_cls(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        collate_fn=_make_collate_fn(tokenizer, config.max_length, torch),
        generator=generator if shuffle else None,
    )


def _spam_probabilities(model: Any, loader: Any, device: str, torch: Any) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            batch.pop("labels")
            inputs = {key: value.to(device) for key, value in batch.items()}
            logits = model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)[:, 1]
            chunks.append(probabilities.detach().cpu().numpy())
    if not chunks:
        return np.asarray([], dtype=float)
    return np.concatenate(chunks).astype(float)


def _train_epoch(
    model: Any,
    loader: Any,
    optimizer: Any,
    class_weights: Any,
    device: str,
    torch: Any,
) -> float:
    model.train()
    running_loss = 0.0
    observations = 0
    for batch in loader:
        labels = batch.pop("labels").to(device)
        inputs = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        logits = model(**inputs).logits
        loss = torch.nn.functional.cross_entropy(logits, labels, weight=class_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        batch_size = int(labels.shape[0])
        running_loss += float(loss.detach().cpu()) * batch_size
        observations += batch_size
    return running_loss / max(observations, 1)


def _artifact_manifest(directory: Path) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.name == "m5_manifest.json":
            continue
        relative = path.relative_to(directory).as_posix()
        files[relative] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    payload = {"files": files}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["artifact_fingerprint"] = hashlib.sha256(canonical).hexdigest()
    return payload


def verify_artifact_manifest(directory: Path) -> bool:
    manifest_path = directory / "m5_manifest.json"
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("files", {})
    for relative, metadata in expected.items():
        path = directory / relative
        if not path.exists() or sha256_file(path) != metadata.get("sha256"):
            return False
    current = {"files": expected}
    canonical = json.dumps(current, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest() == manifest.get("artifact_fingerprint")


def _load_baselines(paths: ProjectPaths) -> dict[str, Any]:
    baselines: dict[str, Any] = {}
    m3_path = paths.reports / "m3_results.json"
    if m3_path.exists():
        m3 = json.loads(m3_path.read_text(encoding="utf-8"))
        baselines["m3"] = {
            "selected_model": m3.get("selected_model"),
            "test_metrics": m3.get("test_metrics"),
        }
    m4_path = paths.reports / "m4_results.json"
    if m4_path.exists():
        m4 = json.loads(m4_path.read_text(encoding="utf-8"))
        baselines["m4"] = {
            "selected_threshold": m4.get("selected_threshold"),
            "test_probability_metrics": m4.get("test_probability_metrics"),
            "test_decision_metrics": m4.get("test_decision_metrics"),
        }
    return baselines


def _write_report(payload: dict[str, Any], path: Path) -> None:
    test = payload["test_metrics"]
    prob = payload["test_probability_metrics"]
    lines = [
        "# M5 Transformer Benchmark",
        "",
        "M5 fine-tunes a compact pretrained Transformer on the frozen M1 train split. "
        "Epoch selection uses validation PR-AUC only; the previously reported M1 test split is "
        "used as a frozen benchmark rather than a pristine final lockbox.",
        "",
        "## Training configuration",
        "",
        f"- Model: `{payload['config']['model_name']}`",
        f"- Device: `{payload['device']}`",
        f"- Epoch budget: {payload['config']['epochs']}",
        f"- Best epoch: {payload['best_epoch']}",
        f"- Batch size: {payload['config']['batch_size']}",
        f"- Max token length: {payload['config']['max_length']}",
        f"- Learning rate: {payload['config']['learning_rate']}",
        f"- Weighted cross-entropy class weights: {payload['class_weights']}",
        "",
        "## Training history",
        "",
        "| epoch | train loss | validation precision | recall | F1 | PR-AUC | ROC-AUC | FP | FN |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["history"]:
        lines.append(
            f"| {row['epoch']} | {row['train_loss']:.4f} | {row['validation_precision']:.4f} | "
            f"{row['validation_recall']:.4f} | {row['validation_f1']:.4f} | "
            f"{row['validation_pr_auc']:.4f} | {row['validation_roc_auc']:.4f} | "
            f"{row['validation_fp']} | {row['validation_fn']} |"
        )
    lines.extend(
        [
            "",
            "## Frozen test benchmark",
            "",
            f"- Precision: {test['precision']:.4f}",
            f"- Recall: {test['recall']:.4f}",
            f"- F1: {test['f1']:.4f}",
            f"- PR-AUC: {test['pr_auc']:.4f}",
            f"- ROC-AUC: {test['roc_auc']:.4f}",
            (
                f"- Confusion matrix: TN={test['tn']}, FP={test['fp']}, "
                f"FN={test['fn']}, TP={test['tp']}"
            ),
            f"- Brier score: {prob['brier']:.4f}",
            f"- Log loss: {prob['log_loss']:.4f}",
            f"- ECE: {prob['ece']:.4f}",
        ]
    )
    m3 = payload.get("baselines", {}).get("m3")
    if m3 and m3.get("test_metrics"):
        sparse = m3["test_metrics"]
        lines.extend(
            [
                "",
                "## Comparison with M3 sparse champion",
                "",
                f"- M3 model: `{m3.get('selected_model')}`",
                f"- M3 test PR-AUC: {sparse['pr_auc']:.4f}",
                f"- M5 test PR-AUC: {test['pr_auc']:.4f}",
                f"- PR-AUC delta: {test['pr_auc'] - sparse['pr_auc']:+.4f}",
                f"- M3 test F1: {sparse['f1']:.4f}",
                f"- M5 test F1: {test['f1']:.4f}",
                f"- F1 delta: {test['f1'] - sparse['f1']:+.4f}",
                f"- M3 FP/FN: {sparse['fp']}/{sparse['fn']}",
                f"- M5 FP/FN: {test['fp']}/{test['fn']}",
            ]
        )
    lines.extend(
        [
            "",
            "## Methodological notes",
            "",
            (
                "- Pretrained-model weights are downloaded from Hugging Face on the first M5 "
                "run and then cached locally."
            ),
            (
                "- MPS is preferred automatically on supported Apple Silicon; CUDA and CPU "
                "fallbacks are supported."
            ),
            (
                "- M5 does not re-tune the M4 business threshold. It is a model-quality "
                "benchmark at probability 0.5."
            ),
            "- M6 introduces external email-domain evidence before any production claim.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_m5(paths: ProjectPaths, config: M5Config | None = None) -> dict[str, Any]:
    config = config or M5Config()
    config.validate()
    paths.ensure_runtime_dirs()
    torch, data_loader_cls, dataset_base, hf_classes = _require_transformer_stack()
    auto_tokenizer, auto_model = hf_classes
    device = resolve_device(config.device)
    _seed_everything(config.random_state, torch)

    train = _load_split(paths.data_processed / "train.csv")
    validation = _load_split(paths.data_processed / "validation.csv")
    test = _load_split(paths.data_processed / "test.csv")

    tokenizer = auto_tokenizer.from_pretrained(config.model_name)
    model = auto_model.from_pretrained(
        config.model_name,
        num_labels=2,
        id2label={0: "ham", 1: "spam"},
        label2id={"ham": 0, "spam": 1},
    )
    model.to(device)

    train_loader = _make_loader(
        train, tokenizer, config, torch, data_loader_cls, dataset_base, shuffle=True
    )
    validation_loader = _make_loader(
        validation, tokenizer, config, torch, data_loader_cls, dataset_base, shuffle=False
    )
    test_loader = _make_loader(
        test, tokenizer, config, torch, data_loader_cls, dataset_base, shuffle=False
    )

    class_weights_np = compute_class_weights(train["label"].astype(int).to_numpy())
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    artifact_dir = paths.artifacts / "m5_transformer"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_pr_auc = -np.inf
    stale_epochs = 0

    y_validation = validation["label"].astype(int).to_numpy()
    for epoch in range(1, config.epochs + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, class_weights, device, torch)
        validation_prob = _spam_probabilities(model, validation_loader, device, torch)
        metrics = transformer_metrics(y_validation, validation_prob)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"validation_{key}": value for key, value in metrics.items()},
        }
        history.append(row)
        if metrics["pr_auc"] > best_pr_auc + 1e-12:
            best_pr_auc = float(metrics["pr_auc"])
            stale_epochs = 0
            model.save_pretrained(artifact_dir, safe_serialization=True)
            tokenizer.save_pretrained(artifact_dir)
        else:
            stale_epochs += 1
            if stale_epochs > config.patience:
                break

    best = select_best_epoch(history)
    model = auto_model.from_pretrained(artifact_dir, local_files_only=True)
    model.to(device)
    test_prob = _spam_probabilities(model, test_loader, device, torch)
    y_test = test["label"].astype(int).to_numpy()
    test_metrics = transformer_metrics(y_test, test_prob)
    test_probability = probability_metrics(y_test, test_prob, bins=config.calibration_bins)

    history_path = paths.reports / "m5_training_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    payload: dict[str, Any] = {
        "milestone": "M5",
        "config": asdict(config),
        "device": device,
        "class_weights": [float(value) for value in class_weights_np],
        "best_epoch": int(best["epoch"]),
        "best_validation_pr_auc": float(best["validation_pr_auc"]),
        "history": history,
        "test_metrics": test_metrics,
        "test_probability_metrics": test_probability,
        "test_ece_recomputed": expected_calibration_error(
            y_test, test_prob, bins=config.calibration_bins
        ),
        "baselines": _load_baselines(paths),
    }
    results_path = paths.reports / "m5_results.json"
    results_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_report(payload, paths.reports / "m5_report.md")

    manifest = _artifact_manifest(artifact_dir)
    manifest["metadata"] = {
        "milestone": "M5",
        "model_name": config.model_name,
        "best_epoch": int(best["epoch"]),
        "test_pr_auc": float(test_metrics["pr_auc"]),
    }
    (artifact_dir / "m5_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def load_m5_for_inference(
    paths: ProjectPaths, requested_device: str = "auto"
) -> tuple[Any, Any, str]:
    torch, _, _, hf_classes = _require_transformer_stack()
    auto_tokenizer, auto_model = hf_classes
    artifact_dir = paths.artifacts / "m5_transformer"
    if not artifact_dir.exists():
        raise FileNotFoundError("M5 transformer artifact not found. Run `make m5-train` first.")
    if not verify_artifact_manifest(artifact_dir):
        raise RuntimeError("M5 transformer artifact manifest verification failed")
    device = resolve_device(requested_device)
    tokenizer = auto_tokenizer.from_pretrained(artifact_dir, local_files_only=True)
    model = auto_model.from_pretrained(artifact_dir, local_files_only=True)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def predict_m5_probabilities(
    texts: list[str], tokenizer: Any, model: Any, device: str, max_length: int = 128
) -> np.ndarray:
    torch, _, _, _ = _require_transformer_stack()
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        logits = model(**encoded).logits
        return torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy().astype(float)
