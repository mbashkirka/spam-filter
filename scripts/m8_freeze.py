from pathlib import Path

from spam_filter.config import ProjectPaths
from spam_filter.m8 import build_m8_serving_artifact


def main() -> int:
    payload = build_m8_serving_artifact(ProjectPaths.from_root(Path.cwd()))
    print(f"model_version={payload['model_version']}")
    print(f"threshold={payload['serving_policy']['threshold']:.6f}")
    print(f"artifact_fingerprint={payload['artifact_fingerprint']}")
    print("milestone=M8")
    print("status=frozen")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
