import json
from pathlib import Path

import numpy as np

from spam_filter.config import ProjectPaths
from spam_filter.data import sha256_file
from spam_filter.m4 import load_m4_model


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    paths = ProjectPaths.from_root(root)
    report_path = paths.reports / "m4_results.json"
    artifact_path = paths.artifacts / "m4_calibrated_champion.joblib"
    if not report_path.exists() or not artifact_path.exists():
        raise FileNotFoundError("M4 outputs missing. Run `make m4-train` first.")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_hash = report["artifact"]["sha256"]
    if sha256_file(artifact_path) != expected_hash:
        raise RuntimeError("M4 artifact hash mismatch")

    model = load_m4_model(artifact_path)
    examples = [
        "Can we meet for lunch tomorrow at 12?",
        "URGENT! You won a FREE prize. Claim now at http://winner.example",
    ]
    probabilities = model.predict_proba(examples)[:, 1]
    if not np.all((probabilities >= 0) & (probabilities <= 1)):
        raise RuntimeError("M4 probabilities are outside [0, 1]")
    if probabilities[1] <= probabilities[0]:
        raise RuntimeError("M4 smoke examples are not ranked in the expected order")

    print(f"selected_model={model.selected_model_name}")
    print(f"selected_threshold={model.threshold:.6f}")
    print(f"ham_example_spam_probability={probabilities[0]:.4f}")
    print(f"spam_example_spam_probability={probabilities[1]:.4f}")
    print("probability_order=ok")
    print("artifact_hash=ok")
    print("milestone=M4")
    print("status=ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
