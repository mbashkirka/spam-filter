from __future__ import annotations

import logging
import os
import time
import uuid
from pathlib import Path
from threading import Lock
from typing import Literal

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from spam_filter.config import ProjectPaths
from spam_filter.m8 import API_VERSION, MODEL_VERSION, FrozenSpamService
from spam_filter.m9 import (
    RuntimeMetrics,
    drift_snapshot,
    load_reference_profile,
    log_prediction_event,
)

logger = logging.getLogger("spam_filter.prediction")


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4096)
    channel: Literal["sms", "email", "unknown"] = "unknown"


class ServiceState:
    def __init__(self, root: Path) -> None:
        self.paths = ProjectPaths.from_root(root)
        self._service: FrozenSpamService | None = None
        self._error: str | None = None
        self._lock = Lock()
        self.metrics = RuntimeMetrics()
        self.reference_profile = load_reference_profile(self.paths)

    def service(self) -> FrozenSpamService:
        if self._service is not None:
            return self._service
        with self._lock:
            if self._service is None:
                try:
                    self._service = FrozenSpamService.load(self.paths)
                    self._error = None
                except Exception as exc:
                    self._error = f"{type(exc).__name__}: {exc}"
                    raise
        return self._service

    @property
    def error(self) -> str | None:
        return self._error


def _default_root() -> Path:
    return Path(os.environ.get("SPAM_FILTER_ROOT", Path.cwd()))


def create_app(root: str | Path | None = None) -> FastAPI:
    resolved_root = Path(root) if root is not None else _default_root()
    state = ServiceState(resolved_root)
    app = FastAPI(
        title="Spam Filter Service",
        version=MODEL_VERSION,
        description="M9 production API for the frozen English SMS spam policy.",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "milestone": "M9", "api_version": API_VERSION}

    @app.get("/ready")
    def ready() -> dict[str, str]:
        try:
            service = state.service()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"model_not_ready: {exc}") from exc
        return {
            "status": "ready",
            "model_version": service.manifest["model_version"],
            "artifact_fingerprint": service.manifest["artifact_fingerprint"],
        }

    @app.get("/model-info")
    def model_info() -> dict:
        try:
            service = state.service()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"model_not_ready: {exc}") from exc
        manifest = service.manifest
        return {
            "api_version": manifest["api_version"],
            "model_version": manifest["model_version"],
            "artifact_fingerprint": manifest["artifact_fingerprint"],
            "serving_policy": manifest["serving_policy"],
            "intended_use": manifest["intended_use"],
            "in_domain_evidence": manifest["in_domain_evidence"],
            "domain_risk": manifest["domain_risk"],
        }

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(
            content=state.metrics.prometheus_text(), media_type="text/plain; version=0.0.4"
        )

    @app.get("/metrics.json")
    def metrics_json() -> dict:
        return state.metrics.snapshot()

    @app.get("/drift")
    def drift() -> dict:
        return drift_snapshot(state.metrics, state.reference_profile)

    @app.post("/predict")
    @app.post("/v1/predict")
    def predict(request: PredictRequest) -> dict:
        request_id = uuid.uuid4().hex
        start = time.perf_counter_ns()
        try:
            service = state.service()
            result = service.predict_one(request.text)
        except Exception as exc:
            state.metrics.record_error()
            raise HTTPException(status_code=503, detail=f"model_not_ready: {exc}") from exc
        latency_ms = (time.perf_counter_ns() - start) / 1_000_000.0
        result["channel"] = request.channel
        result["input_chars"] = len(request.text)
        warning = request.channel != "sms"
        result["domain_warning"] = service.manifest["domain_risk"]["warning"] if warning else None
        result["request_id"] = request_id
        state.metrics.record_prediction(
            label=result["label"],
            probability=result["spam_probability"],
            input_chars=len(request.text),
            latency_ms=latency_ms,
            domain_warning=warning,
        )
        log_prediction_event(
            logger,
            request_id=request_id,
            channel=request.channel,
            input_chars=len(request.text),
            label=result["label"],
            probability=result["spam_probability"],
            latency_ms=latency_ms,
            domain_warning=warning,
            model_version=result["model_version"],
        )
        return result

    return app


app = create_app()
