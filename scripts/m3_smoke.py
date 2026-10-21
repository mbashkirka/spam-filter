from __future__ import annotations

import json
from pathlib import Path

from spam_filter.config import ProjectPaths
from spam_filter.data import sha256_file
from spam_filter.m3 import _continuous_scores, load_m3_model


def main() -> int:
    paths = ProjectPaths.from_root(Path.cwd())
    report_path = paths.reports / "m3_results.json"
    model_path = paths.artifacts / "m3_sparse_champion.joblib"
    if not report_path.exists() or not model_path.exists():
        raise FileNotFoundError("M3 artifacts missing. Run `make m3-train` first.")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    actual_hash = sha256_file(model_path)
    expected_hash = report["artifact"]["sha256"]
    if actual_hash != expected_hash:
        raise RuntimeError("M3 artifact SHA-256 does not match report")

    model = load_m3_model(model_path)
    examples = [
        "Can we meet at the station at 7 tonight?",
        "WIN a guaranteed cash prize now, claim immediately at www.example.com",
    ]
    predictions = model.predict(examples).astype(int).tolist()
    scores = _continuous_scores(model, examples).tolist()

    print(f"selected_model={report['selected_model']}")
    print(f"locked_test_pr_auc={report['test_metrics']['pr_auc']:.4f}")
    print(f"ham_example_prediction={predictions[0]}")
    print(f"spam_example_prediction={predictions[1]}")
    print(f"ham_example_score={scores[0]:.4f}")
    print(f"spam_example_score={scores[1]:.4f}")
    print("artifact_hash=ok")
    print("milestone=M3")
    print("status=ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
