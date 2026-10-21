from __future__ import annotations

import hashlib
import json
import logging
import math
import platform
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from threading import Lock
from typing import Any

import numpy as np

from spam_filter import __version__
from spam_filter.config import ProjectPaths
from spam_filter.m8 import (
    MODEL_VERSION,
    FrozenSpamService,
    load_serving_manifest,
    verify_m8_artifact,
)

MILESTONE = "M9"
RELEASE_VERSION = "1.0.0"
LATENCY_REPORT = "m9_latency.json"
RELEASE_REPORT = "m9_release.json"
REFERENCE_PROFILE = "m9_reference_profile.json"


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=float), q))


@dataclass(slots=True)
class RuntimeMetrics:
    """Privacy-safe in-memory observability for one API process."""

    max_latency_samples: int = 10_000
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _requests: int = 0
    _spam: int = 0
    _ham: int = 0
    _domain_warnings: int = 0
    _errors: int = 0
    _input_chars_sum: int = 0
    _probability_sum: float = 0.0
    _latencies_ms: deque[float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._latencies_ms = deque(maxlen=self.max_latency_samples)

    def record_prediction(
        self,
        *,
        label: str,
        probability: float,
        input_chars: int,
        latency_ms: float,
        domain_warning: bool,
    ) -> None:
        with self._lock:
            self._requests += 1
            if label == "spam":
                self._spam += 1
            else:
                self._ham += 1
            self._domain_warnings += int(domain_warning)
            self._input_chars_sum += int(input_chars)
            self._probability_sum += float(probability)
            self._latencies_ms.append(float(latency_ms))

    def record_error(self) -> None:
        with self._lock:
            self._errors += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            latencies = list(self._latencies_ms)
            requests = self._requests
            return {
                "requests_total": requests,
                "predictions": {"spam": self._spam, "ham": self._ham},
                "errors_total": self._errors,
                "domain_warnings_total": self._domain_warnings,
                "domain_warning_rate": self._domain_warnings / requests if requests else 0.0,
                "predicted_spam_rate": self._spam / requests if requests else 0.0,
                "mean_spam_probability": self._probability_sum / requests if requests else 0.0,
                "mean_input_chars": self._input_chars_sum / requests if requests else 0.0,
                "latency_ms": {
                    "samples": len(latencies),
                    "mean": mean(latencies) if latencies else 0.0,
                    "p50": _quantile(latencies, 0.50),
                    "p95": _quantile(latencies, 0.95),
                    "p99": _quantile(latencies, 0.99),
                },
            }

    def prometheus_text(self) -> str:
        s = self.snapshot()
        lines = [
            "# HELP spam_filter_requests_total Prediction requests served.",
            "# TYPE spam_filter_requests_total counter",
            f"spam_filter_requests_total {s['requests_total']}",
            "# HELP spam_filter_predictions_total Predictions by output label.",
            "# TYPE spam_filter_predictions_total counter",
            f"spam_filter_predictions_total{{label=\"spam\"}} {s['predictions']['spam']}",
            f"spam_filter_predictions_total{{label=\"ham\"}} {s['predictions']['ham']}",
            "# HELP spam_filter_domain_warnings_total Out-of-domain request warnings.",
            "# TYPE spam_filter_domain_warnings_total counter",
            f"spam_filter_domain_warnings_total {s['domain_warnings_total']}",
            "# HELP spam_filter_errors_total Prediction errors.",
            "# TYPE spam_filter_errors_total counter",
            f"spam_filter_errors_total {s['errors_total']}",
            "# HELP spam_filter_latency_ms_p95 Process-local prediction latency p95.",
            "# TYPE spam_filter_latency_ms_p95 gauge",
            f"spam_filter_latency_ms_p95 {s['latency_ms']['p95']:.6f}",
            "# HELP spam_filter_mean_input_chars Mean request length in characters.",
            "# TYPE spam_filter_mean_input_chars gauge",
            f"spam_filter_mean_input_chars {s['mean_input_chars']:.6f}",
            "# HELP spam_filter_predicted_spam_rate Fraction predicted spam.",
            "# TYPE spam_filter_predicted_spam_rate gauge",
            f"spam_filter_predicted_spam_rate {s['predicted_spam_rate']:.6f}",
        ]
        return "\n".join(lines) + "\n"


def load_reference_profile(paths: ProjectPaths) -> dict[str, float]:
    profile_path = paths.artifacts / REFERENCE_PROFILE
    if profile_path.exists():
        return json.loads(profile_path.read_text(encoding="utf-8"))
    eda_path = paths.reports / "m1_eda.json"
    if not eda_path.exists():
        return {"spam_rate": 0.1263, "median_chars": 61.0, "p95_chars": 160.0}
    eda = json.loads(eda_path.read_text(encoding="utf-8"))
    return {
        "spam_rate": float(eda["spam_rate"]),
        "median_chars": float(eda["text_length_chars"]["median"]),
        "p95_chars": float(eda["text_length_chars"]["p95"]),
    }


def write_reference_profile(paths: ProjectPaths) -> dict[str, float]:
    paths.ensure_runtime_dirs()
    profile = load_reference_profile(paths)
    target = paths.artifacts / REFERENCE_PROFILE
    target.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return profile


def drift_snapshot(metrics: RuntimeMetrics, reference: dict[str, float]) -> dict[str, Any]:
    current = metrics.snapshot()
    n = int(current["requests_total"])
    if n < 20:
        return {
            "status": "insufficient_observations",
            "observations": n,
            "minimum_observations": 20,
            "reference": reference,
            "signals": {},
        }
    warning_rate = float(current["domain_warning_rate"])
    length_ratio = float(current["mean_input_chars"]) / max(float(reference["p95_chars"]), 1.0)
    spam_delta = abs(float(current["predicted_spam_rate"]) - float(reference["spam_rate"]))
    signals = {
        "out_of_domain_rate": {"value": warning_rate, "alert": warning_rate > 0.05},
        "mean_length_vs_reference_p95": {"value": length_ratio, "alert": length_ratio > 1.25},
        "predicted_spam_rate_abs_delta": {"value": spam_delta, "alert": spam_delta > 0.15},
    }
    return {
        "status": "alert" if any(v["alert"] for v in signals.values()) else "ok",
        "observations": n,
        "reference": reference,
        "signals": signals,
        "note": "Unsupervised process-level drift signals; labels are not inferred online.",
    }


def log_prediction_event(
    logger: logging.Logger,
    *,
    request_id: str,
    channel: str,
    input_chars: int,
    label: str,
    probability: float,
    latency_ms: float,
    domain_warning: bool,
    model_version: str,
) -> None:
    payload = {
        "event": "spam_prediction",
        "request_id": request_id,
        "channel": channel,
        "input_chars": int(input_chars),
        "label": label,
        "spam_probability": round(float(probability), 6),
        "latency_ms": round(float(latency_ms), 3),
        "domain_warning": bool(domain_warning),
        "model_version": model_version,
    }
    logger.info(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def benchmark_service(
    service: FrozenSpamService,
    *,
    iterations: int = 500,
    warmup: int = 25,
    slo_p95_ms: float = 50.0,
) -> dict[str, Any]:
    if iterations < 10 or warmup < 0 or slo_p95_ms <= 0:
        raise ValueError("Invalid benchmark configuration")
    samples = [
        "Are we still meeting at 7 tonight?",
        "WIN a FREE prize now. Claim your reward today!",
        "Please call me when you arrive.",
        "URGENT! You have won a cash reward. Reply now to claim.",
    ]
    for i in range(warmup):
        service.predict_one(samples[i % len(samples)])
    latencies: list[float] = []
    start_all = time.perf_counter()
    for i in range(iterations):
        start = time.perf_counter_ns()
        service.predict_one(samples[i % len(samples)])
        latencies.append((time.perf_counter_ns() - start) / 1_000_000.0)
    elapsed = time.perf_counter() - start_all
    p95 = _quantile(latencies, 0.95)
    return {
        "benchmark": "local_model_inference",
        "iterations": iterations,
        "warmup": warmup,
        "latency_ms": {
            "mean": mean(latencies),
            "p50": _quantile(latencies, 0.50),
            "p95": p95,
            "p99": _quantile(latencies, 0.99),
            "max": max(latencies),
        },
        "throughput_predictions_per_second": iterations / elapsed if elapsed > 0 else math.inf,
        "slo": {"p95_ms_max": slo_p95_ms, "passed": p95 <= slo_p95_ms},
        "model_version": service.manifest["model_version"],
        "artifact_fingerprint": service.manifest["artifact_fingerprint"],
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


def write_latency_report(paths: ProjectPaths, result: dict[str, Any]) -> Path:
    paths.ensure_runtime_dirs()
    path = paths.reports / LATENCY_REPORT
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_release_contract(
    paths: ProjectPaths, *, strict_python: bool = True
) -> dict[str, Any]:
    if __version__ != RELEASE_VERSION:
        raise RuntimeError(f"Package version must be {RELEASE_VERSION}, got {__version__}")
    python_supported = sys.version_info[:2] == (3, 12)
    if strict_python and not python_supported:
        raise RuntimeError("M9 release is validated on Python 3.12")
    if not verify_m8_artifact(paths):
        raise RuntimeError("M8 serving artifact is missing or invalid")
    latency_path = paths.reports / LATENCY_REPORT
    if not latency_path.exists():
        raise FileNotFoundError("Run `make m9-benchmark` before `make m9-release`")
    latency = json.loads(latency_path.read_text(encoding="utf-8"))
    if not latency["slo"]["passed"]:
        raise RuntimeError("Latency SLO did not pass")
    required = [
        paths.root / "MODEL_CARD.md",
        paths.root / "Dockerfile",
        paths.root / ".dockerignore",
        paths.root / ".github" / "workflows" / "ci.yml",
        paths.reports / "m8_report.md",
        paths.reports / "m6_results.json",
        paths.reports / "m7_results.json",
    ]
    missing = [str(path.relative_to(paths.root)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing M9 release inputs: {missing}")
    profile = write_reference_profile(paths)
    manifest = load_serving_manifest(paths)
    payload = {
        "milestone": MILESTONE,
        "release_version": RELEASE_VERSION,
        "model_version": MODEL_VERSION,
        "artifact_fingerprint": manifest["artifact_fingerprint"],
        "release_status": "ready",
        "python": platform.python_version(),
        "python_supported": python_supported,
        "platform": platform.platform(),
        "latency": latency,
        "reference_profile": profile,
        "privacy": {
            "raw_text_logged": False,
            "online_signals": [
                "request count",
                "channel",
                "input length",
                "prediction label",
                "spam probability",
                "latency",
                "domain warning",
            ],
        },
        "intended_use": manifest["intended_use"],
        "domain_risk": manifest["domain_risk"],
        "release_files": {
            str(path.relative_to(paths.root)): _file_sha256(path) for path in required[:4]
        },
    }
    path = paths.reports / RELEASE_REPORT
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = paths.reports / "m9_report.md"
    latency_ms = latency["latency_ms"]
    md.write_text(
        "\n".join(
            [
                "# M9 Production Release Report",
                "",
                f"- Release: `{RELEASE_VERSION}` / `{MILESTONE}`",
                f"- Model: `{MODEL_VERSION}`",
                f"- Artifact fingerprint: `{manifest['artifact_fingerprint']}`",
                "- Release status: `ready`",
                "",
                "## Local inference performance",
                "",
                f"- Mean: {latency_ms['mean']:.3f} ms",
                f"- p50: {latency_ms['p50']:.3f} ms",
                f"- p95: {latency_ms['p95']:.3f} ms",
                f"- p99: {latency_ms['p99']:.3f} ms",
                f"- Throughput: {latency['throughput_predictions_per_second']:.1f} predictions/s",
                f"- Declared local p95 SLO: <= {latency['slo']['p95_ms_max']:.1f} ms — **PASS**",
                "",
                "## Observability & privacy",
                "",
                "- `/metrics` exposes process-local aggregate metrics in Prometheus text format.",
                "- `/metrics.json` exposes the same aggregate snapshot as JSON.",
                "- `/drift` reports unsupervised domain/length/prediction-rate drift signals.",
                "- Structured prediction logs never include raw message text.",
                "- Metrics are process-local and reset on restart; production deployment "
                "should scrape them.",
                "",
                "## Release boundaries",
                "",
                "- Intended use remains English SMS-like text.",
                "- Email remains confirmed high-risk out-of-domain based on M6/M7.",
                "- Online drift signals do not substitute for labeled monitoring or "
                "retraining validation.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload
