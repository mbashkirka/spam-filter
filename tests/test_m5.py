from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from spam_filter.m5 import (
    M5Config,
    compute_class_weights,
    select_best_epoch,
    transformer_metrics,
    verify_artifact_manifest,
)


def test_m5_config_validation() -> None:
    M5Config().validate()
    with pytest.raises(ValueError):
        M5Config(epochs=0).validate()
    with pytest.raises(ValueError):
        M5Config(device="tpu").validate()


def test_class_weights_balance_inverse_frequency() -> None:
    weights = compute_class_weights([0, 0, 0, 1])
    assert weights[1] > weights[0]
    assert np.allclose(weights, [4 / 6, 2.0])


def test_class_weights_require_both_classes() -> None:
    with pytest.raises(ValueError):
        compute_class_weights([0, 0, 0])


def test_transformer_metrics_contract() -> None:
    metrics = transformer_metrics(
        np.array([0, 0, 1, 1], dtype=np.int8),
        np.array([0.1, 0.4, 0.6, 0.9]),
    )
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    assert metrics["pr_auc"] == pytest.approx(1.0)


def test_select_best_epoch_uses_validation_pr_auc() -> None:
    history = [
        {"epoch": 1, "validation_pr_auc": 0.90},
        {"epoch": 2, "validation_pr_auc": 0.95},
        {"epoch": 3, "validation_pr_auc": 0.94},
    ]
    assert select_best_epoch(history)["epoch"] == 2


def test_manifest_verification_detects_tampering(tmp_path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    model = artifact / "model.safetensors"
    model.write_bytes(b"weights")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    payload = {"files": {"model.safetensors": {"sha256": digest, "bytes": 7}}}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["artifact_fingerprint"] = hashlib.sha256(canonical).hexdigest()
    (artifact / "m5_manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    assert verify_artifact_manifest(artifact)
    model.write_bytes(b"tampered")
    assert not verify_artifact_manifest(artifact)
