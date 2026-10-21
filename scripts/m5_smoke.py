from __future__ import annotations

import json
from pathlib import Path

from spam_filter.config import ProjectPaths
from spam_filter.m5 import (
    load_m5_for_inference,
    predict_m5_probabilities,
    verify_artifact_manifest,
)


def main() -> int:
    paths = ProjectPaths.from_root(Path.cwd())
    results_path = paths.reports / "m5_results.json"
    if not results_path.exists():
        raise FileNotFoundError("M5 results not found. Run `make m5-train` first.")
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    tokenizer, model, device = load_m5_for_inference(paths)
    examples = [
        "Are we still meeting for lunch at 12 tomorrow?",
        "URGENT! You have won a guaranteed cash prize. Claim now!",
    ]
    probabilities = predict_m5_probabilities(examples, tokenizer, model, device)
    if probabilities.shape != (2,):
        raise RuntimeError("Unexpected M5 inference shape")
    if not probabilities[1] > probabilities[0]:
        raise RuntimeError("M5 smoke examples are not ordered as expected")
    artifact_ok = verify_artifact_manifest(paths.artifacts / "m5_transformer")
    if not artifact_ok:
        raise RuntimeError("M5 artifact hash verification failed")
    print(f"model_name={payload['config']['model_name']}")
    print(f"device={device}")
    print(f"best_epoch={payload['best_epoch']}")
    print(f"locked_test_pr_auc={payload['test_metrics']['pr_auc']:.4f}")
    print(f"ham_example_spam_probability={probabilities[0]:.4f}")
    print(f"spam_example_spam_probability={probabilities[1]:.4f}")
    print("probability_order=ok")
    print("artifact_hash=ok")
    print("milestone=M5")
    print("status=ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
