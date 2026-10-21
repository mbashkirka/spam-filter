from __future__ import annotations

from pathlib import Path

from spam_filter.config import ProjectPaths
from spam_filter.m6 import M6Config, run_m6


def main() -> int:
    paths = ProjectPaths.from_root(Path.cwd())
    result = run_m6(paths, M6Config())
    m3 = result["models"]["m3_sparse"]
    m4 = result["models"]["m4_calibrated_sparse"]
    m5 = result["models"]["m5_transformer"]
    print(f"rows={result['dataset']['rows_unique']}")
    print(f"m3_pr_auc={m3['pr_auc']:.4f}")
    print(f"m3_f1={m3['f1']:.4f}")
    print(f"m4_pr_auc={m4['pr_auc']:.4f}")
    print(f"m4_f1={m4['f1']:.4f}")
    print(f"m5_pr_auc={m5['pr_auc']:.4f}")
    print(f"m5_f1={m5['f1']:.4f}")
    print(f"device={m5['device']}")
    print("milestone=M6")
    print("status=evaluated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
