from pathlib import Path

from spam_filter.config import ProjectPaths
from spam_filter.m7 import M7Config, run_m7


def main() -> int:
    result = run_m7(ProjectPaths.from_root(Path.cwd()), M7Config())
    agreement = result["external"]["agreement"]
    print(f"enron_rows={result['external']['rows']}")
    print(f"both_wrong={agreement['both_wrong']}")
    print(f"only_m3_correct={agreement['only_m3_correct']}")
    print(f"only_m5_correct={agreement['only_m5_correct']}")
    print(f"stress_spam_rows={result['robustness']['spam_rows']}")
    print(f"device={result['robustness']['device']}")
    print("milestone=M7")
    print("status=analyzed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
