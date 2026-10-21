from __future__ import annotations

import json
from pathlib import Path

from spam_filter.config import ProjectPaths
from spam_filter.data import sha256_file
from spam_filter.naive_bayes import ScratchMultinomialNB, spam_probability


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    paths = ProjectPaths.from_root(root)
    model_path = paths.artifacts / "m2_scratch_nb.json"
    report_path = paths.reports / "m2_results.json"
    if not model_path.exists() or not report_path.exists():
        raise FileNotFoundError("M2 artifacts are missing. Run `make m2-train` first.")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("milestone") != "M2":
        raise AssertionError("Unexpected M2 report milestone")
    if report["artifact"]["sha256"] != sha256_file(model_path):
        raise AssertionError("M2 model artifact hash mismatch")
    if len(report["validation_results"]) != 4:
        raise AssertionError("Expected four preprocessing ablations")

    model, preprocessor = ScratchMultinomialNB.load(model_path)
    ham_probability = spam_probability(
        model, preprocessor, "Are we still meeting for lunch tomorrow?"
    )
    spam_probability_value = spam_probability(
        model, preprocessor, "WIN cash prize now! Call 0800123456 or visit www.claim-prize.com"
    )
    for value in (ham_probability, spam_probability_value):
        if not 0.0 <= value <= 1.0:
            raise AssertionError("Probability is outside [0, 1]")

    print(f"selected_variant={report['selected']['variant']}")
    print(f"vocabulary_size={report['selected']['vocabulary_size']}")
    print(f"locked_test_pr_auc={report['test_metrics']['pr_auc']:.4f}")
    print(f"ham_example_spam_probability={ham_probability:.4f}")
    print(f"spam_example_spam_probability={spam_probability_value:.4f}")
    print("artifact_hash=ok")
    print("milestone=M2")
    print("status=ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
