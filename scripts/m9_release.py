from __future__ import annotations

from pathlib import Path

from spam_filter.config import ProjectPaths
from spam_filter.m9 import build_release_contract


def main() -> int:
    result = build_release_contract(ProjectPaths.from_root(Path.cwd()))
    print(f"release_version={result['release_version']}")
    print(f"model_version={result['model_version']}")
    print(f"artifact_fingerprint={result['artifact_fingerprint']}")
    print('release_status=ready')
    print('milestone=M9')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
