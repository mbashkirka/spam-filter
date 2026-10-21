from __future__ import annotations

from pathlib import Path

from spam_filter.config import ProjectPaths
from spam_filter.m5 import M5Config, run_m5


def main() -> int:
    result = run_m5(ProjectPaths.from_root(Path.cwd()), M5Config())
    test = result["test_metrics"]
    print(f"model_name={result['config']['model_name']}")
    print(f"device={result['device']}")
    print(f"best_epoch={result['best_epoch']}")
    print(f"validation_pr_auc={result['best_validation_pr_auc']:.4f}")
    print(f"test_pr_auc={test['pr_auc']:.4f}")
    print(f"test_f1={test['f1']:.4f}")
    print(f"test_fp={test['fp']}")
    print(f"test_fn={test['fn']}")
    print("milestone=M5")
    print("status=trained")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
