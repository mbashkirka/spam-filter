from pathlib import Path

from spam_filter.config import ProjectPaths
from spam_filter.m4 import M4Config, run_m4


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    result = run_m4(ProjectPaths.from_root(root), M4Config(fp_cost=10.0, fn_cost=1.0))
    selected = result["selected_threshold"]
    test = result["test_decision_metrics"]
    prob = result["test_probability_metrics"]
    print(f"selected_model={result['selected_model']}")
    print(f"selected_threshold={selected['threshold']:.6f}")
    print(f"validation_weighted_cost={selected['total_cost']:.2f}")
    print(f"test_pr_auc={prob['pr_auc']:.4f}")
    print(f"test_f1={test['f1']:.4f}")
    print(f"test_fp={test['fp']}")
    print(f"test_fn={test['fn']}")
    print(f"test_weighted_cost={test['total_cost']:.2f}")
    print("milestone=M4")
    print("status=trained")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
