from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from spam_filter.config import ProjectPaths
from spam_filter.data import sha256_file
from spam_filter.m4 import CalibratedSpamModel

MODEL_VERSION = "spam-sms-sparse-calibrated-v1"
API_VERSION = "v1"
SERVING_DIRNAME = "m8_serving"
MANIFEST_NAME = "manifest.json"
MODEL_NAME = "model.joblib"


@dataclass(frozen=True, slots=True)
class M8Config:
    intended_channel: str = "sms"
    intended_language: str = "English"
    max_request_chars: int = 4096

    def validate(self) -> None:
        if self.intended_channel != "sms":
            raise ValueError("M8 serving policy is intentionally frozen to the SMS domain")
        if not self.intended_language.strip():
            raise ValueError("intended_language cannot be empty")
        if self.max_request_chars < 160:
            raise ValueError("max_request_chars must be >= 160")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required report not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _safe_metric(mapping: dict[str, Any], key: str) -> float | None:
    value = mapping.get(key)
    return None if value is None else float(value)


def _domain_risk_summary(m6: dict[str, Any], m7: dict[str, Any]) -> dict[str, Any]:
    m4_external = m6["models"]["m4_calibrated_sparse"]
    m5_external = m6["models"]["m5_transformer"]
    agreement = m7["external"]["agreement"]
    return {
        "status": "confirmed_high_risk_out_of_domain",
        "benchmark": "AUEB Enron1 email, zero-shot from SMS training",
        "m4_email_pr_auc": _safe_metric(m4_external, "pr_auc"),
        "m4_email_f1": _safe_metric(m4_external, "f1"),
        "m4_email_ece": _safe_metric(m4_external.get("probability", {}), "ece"),
        "m5_email_pr_auc": _safe_metric(m5_external, "pr_auc"),
        "m5_email_f1": _safe_metric(m5_external, "f1"),
        "m5_email_ece": _safe_metric(m5_external.get("probability", {}), "ece"),
        "both_models_wrong_rows": int(agreement["both_wrong"]),
        "warning": (
            "Do not treat this SMS serving policy as validated for email. M6/M7 demonstrate "
            "severe SMS-to-email domain shift and probability miscalibration."
        ),
    }



def _write_release_report(payload: dict[str, Any], path: Path) -> None:
    policy = payload["serving_policy"]
    evidence = payload["in_domain_evidence"]
    risk = payload["domain_risk"]
    lines = [
        "# M8 Production Serving Report",
        "",
        "M8 freezes the calibrated sparse SMS policy for serving. The release contract",
        "explicitly embeds the M6/M7 email-domain failure evidence.",
        "",
        "## Frozen policy",
        "",
        f"- Model version: `{payload['model_version']}`",
        f"- Model family: {policy['model_family']}",
        f"- Threshold: {policy['threshold']:.6f}",
        f"- FP/FN costs: {policy['fp_cost']:.2f}/{policy['fn_cost']:.2f}",
        f"- Artifact fingerprint: `{payload['artifact_fingerprint']}`",
        "",
        "## In-domain SMS evidence",
        "",
        f"- PR-AUC: {evidence['pr_auc']:.4f}",
        f"- F1: {evidence['f1']:.4f}",
        f"- Precision/recall: {evidence['precision']:.4f}/{evidence['recall']:.4f}",
        f"- FP/FN: {evidence['fp']}/{evidence['fn']}",
        f"- Brier/ECE: {evidence['brier']:.4f}/{evidence['ece']:.4f}",
        "",
        "## Known domain risk",
        "",
        f"- Status: `{risk['status']}`",
        f"- External benchmark: {risk['benchmark']}",
        f"- M4 email PR-AUC/F1/ECE: {risk['m4_email_pr_auc']:.4f}/"
        f"{risk['m4_email_f1']:.4f}/{risk['m4_email_ece']:.4f}",
        f"- M5 email PR-AUC/F1/ECE: {risk['m5_email_pr_auc']:.4f}/"
        f"{risk['m5_email_f1']:.4f}/{risk['m5_email_ece']:.4f}",
        f"- Both M3/M5 wrong on Enron rows: {risk['both_models_wrong_rows']}",
        f"- Warning: {risk['warning']}",
        "",
        "## API",
        "",
        "- `GET /health`",
        "- `GET /ready`",
        "- `GET /model-info`",
        "- `POST /predict`",
        "- `POST /v1/predict`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")

def build_m8_serving_artifact(
    paths: ProjectPaths, config: M8Config | None = None
) -> dict[str, Any]:
    config = config or M8Config()
    config.validate()
    paths.ensure_runtime_dirs()

    source_model = paths.artifacts / "m4_calibrated_champion.joblib"
    m4_path = paths.reports / "m4_results.json"
    m6_path = paths.reports / "m6_results.json"
    m7_path = paths.reports / "m7_results.json"
    required = [source_model, m4_path, m6_path, m7_path]
    missing = [str(path.relative_to(paths.root)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing M8 prerequisites: {missing}")

    m4 = _read_json(m4_path)
    m6 = _read_json(m6_path)
    m7 = _read_json(m7_path)

    expected_source_sha = str(m4["artifact"]["sha256"])
    actual_source_sha = sha256_file(source_model)
    if actual_source_sha != expected_source_sha:
        raise RuntimeError("M4 source artifact hash does not match m4_results.json")

    model = joblib.load(source_model)
    if not isinstance(model, CalibratedSpamModel):
        raise TypeError("M4 serving artifact must be CalibratedSpamModel")

    serving_dir = paths.artifacts / SERVING_DIRNAME
    if serving_dir.exists():
        shutil.rmtree(serving_dir)
    serving_dir.mkdir(parents=True)
    serving_model = serving_dir / MODEL_NAME
    shutil.copy2(source_model, serving_model)

    sms_test = m4["test_decision_metrics"]
    sms_prob = m4["test_probability_metrics"]
    manifest_base: dict[str, Any] = {
        "milestone": "M8",
        "api_version": API_VERSION,
        "model_version": MODEL_VERSION,
        "serving_policy": {
            "model_family": "char TF-IDF + LinearSVC + OOF Platt calibration",
            "source_milestone": "M4",
            "selected_model": str(model.selected_model_name),
            "threshold": float(model.threshold),
            "fp_cost": float(model.fp_cost),
            "fn_cost": float(model.fn_cost),
            "decision_rule": "spam_probability >= threshold",
        },
        "intended_use": {
            "channel": config.intended_channel,
            "language": config.intended_language,
            "description": "English SMS-like spam classification",
            "max_request_chars": int(config.max_request_chars),
            "not_validated_for": ["email", "long-form documents", "non-English text"],
        },
        "in_domain_evidence": {
            "benchmark": "frozen UCI SMS Spam Collection test split",
            "precision": float(sms_test["precision"]),
            "recall": float(sms_test["recall"]),
            "f1": float(sms_test["f1"]),
            "pr_auc": float(sms_prob["pr_auc"]),
            "roc_auc": float(sms_prob["roc_auc"]),
            "brier": float(sms_prob["brier"]),
            "ece": float(sms_prob["ece"]),
            "fp": int(sms_test["fp"]),
            "fn": int(sms_test["fn"]),
        },
        "domain_risk": _domain_risk_summary(m6, m7),
        "robustness": {
            "source": "M7 synthetic frozen-SMS stress test",
            "rows": m7["robustness"]["rows"],
        },
        "artifact": {
            "model_file": MODEL_NAME,
            "model_sha256": sha256_file(serving_model),
            "bytes": serving_model.stat().st_size,
            "source_model_sha256": actual_source_sha,
        },
        "source_reports": {
            "m4_sha256": sha256_file(m4_path),
            "m6_sha256": sha256_file(m6_path),
            "m7_sha256": sha256_file(m7_path),
        },
    }
    manifest_base["artifact_fingerprint"] = _canonical_fingerprint(manifest_base)
    manifest_path = serving_dir / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest_base, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    release_json = paths.reports / "m8_release.json"
    release_md = paths.reports / "m8_report.md"
    release_json.write_text(
        json.dumps(manifest_base, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_release_report(manifest_base, release_md)
    return manifest_base


def load_serving_manifest(paths: ProjectPaths) -> dict[str, Any]:
    path = paths.artifacts / SERVING_DIRNAME / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError("M8 serving manifest missing. Run `make m8-freeze` first.")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_m8_artifact(paths: ProjectPaths) -> bool:
    serving_dir = paths.artifacts / SERVING_DIRNAME
    manifest_path = serving_dir / MANIFEST_NAME
    model_path = serving_dir / MODEL_NAME
    if not manifest_path.exists() or not model_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stored_fingerprint = manifest.pop("artifact_fingerprint", None)
    if not stored_fingerprint or _canonical_fingerprint(manifest) != stored_fingerprint:
        return False
    if sha256_file(model_path) != manifest["artifact"]["model_sha256"]:
        return False
    return True


@dataclass(slots=True)
class FrozenSpamService:
    paths: ProjectPaths
    manifest: dict[str, Any]
    model: CalibratedSpamModel

    @classmethod
    def load(cls, paths: ProjectPaths) -> FrozenSpamService:
        if not verify_m8_artifact(paths):
            raise RuntimeError("M8 serving artifact verification failed")
        manifest = load_serving_manifest(paths)
        model_path = paths.artifacts / SERVING_DIRNAME / MODEL_NAME
        model = joblib.load(model_path)
        if not isinstance(model, CalibratedSpamModel):
            raise TypeError("M8 model is not CalibratedSpamModel")
        if not np.isclose(float(model.threshold), float(manifest["serving_policy"]["threshold"])):
            raise RuntimeError("M8 manifest threshold does not match model artifact")
        return cls(paths=paths, manifest=manifest, model=model)

    def predict_one(self, text: str) -> dict[str, Any]:
        probability = float(self.model.predict_proba([text])[0, 1])
        is_spam = bool(probability >= self.model.threshold)
        return {
            "label": "spam" if is_spam else "ham",
            "is_spam": is_spam,
            "spam_probability": probability,
            "threshold": float(self.model.threshold),
            "model_version": self.manifest["model_version"],
            "artifact_fingerprint": self.manifest["artifact_fingerprint"],
        }


def verify_m8_outputs(paths: ProjectPaths) -> dict[str, bool]:
    return {
        "artifact": verify_m8_artifact(paths),
        "release_json": (paths.reports / "m8_release.json").exists(),
        "report": (paths.reports / "m8_report.md").exists(),
    }
