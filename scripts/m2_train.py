from __future__ import annotations

from pathlib import Path

from spam_filter.config import ProjectPaths
from spam_filter.m2 import run_m2


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    result = run_m2(ProjectPaths.from_root(root))
    selected = result["selected"]
    test = result["test_metrics"]
    print(f"selected_variant={selected['variant']}")
    print(f"validation_pr_auc={selected['pr_auc']:.4f}")
    print(f"test_pr_auc={test['pr_auc']:.4f}")
    print(f"test_f1={test['f1']:.4f}")
    print(f"test_fp={test['fp']}")
    print(f"test_fn={test['fn']}")
    print("milestone=M2")
    print("status=trained")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
