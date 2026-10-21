from __future__ import annotations

import json
from pathlib import Path

from spam_filter.config import ProjectPaths
from spam_filter.data import sha256_file
from spam_filter.m6 import verify_m6_inputs


def main() -> int:
    paths = ProjectPaths.from_root(Path.cwd())
    results_path = paths.reports / "m6_results.json"
    predictions_path = paths.reports / "m6_external_predictions.csv"
    if not results_path.exists():
        raise FileNotFoundError("M6 results not found. Run `make m6-eval` first.")
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    if payload.get("milestone") != "M6":
        raise RuntimeError("Unexpected M6 report milestone")
    if not predictions_path.exists():
        raise FileNotFoundError("M6 prediction audit file is missing")
    if sha256_file(predictions_path) != payload["predictions"]["sha256"]:
        raise RuntimeError("M6 prediction audit hash mismatch")
    inputs = verify_m6_inputs(paths)
    if not all(inputs.values()):
        raise RuntimeError(f"Missing M6 inputs: {inputs}")
    m5 = payload["models"]["m5_transformer"]
    print("dataset=enron1")
    print(f"rows={payload['dataset']['rows_unique']}")
    print(f"m5_external_pr_auc={m5['pr_auc']:.4f}")
    print(f"m5_external_f1={m5['f1']:.4f}")
    print("prediction_audit_hash=ok")
    print("milestone=M6")
    print("status=ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
