from pathlib import Path

from spam_filter.config import ProjectPaths
from spam_filter.m3 import run_m3


def main() -> int:
    result = run_m3(ProjectPaths.from_root(Path.cwd()))
    selected = result["selected_validation"]
    test = result["test_metrics"]
    print(f"selected_model={result['selected_model']}")
    print(f"validation_pr_auc={selected['pr_auc']:.4f}")
    print(f"test_pr_auc={test['pr_auc']:.4f}")
    print(f"test_f1={test['f1']:.4f}")
    print(f"test_fp={test['fp']}")
    print(f"test_fn={test['fn']}")
    print("milestone=M3")
    print("status=trained")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
