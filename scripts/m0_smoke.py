from __future__ import annotations

from pathlib import Path

from spam_filter.cli import main
from spam_filter.config import ProjectPaths


def run() -> int:
    root = Path(__file__).resolve().parents[1]
    paths = ProjectPaths.from_root(root)
    paths.ensure_runtime_dirs()
    return main(["info", "--root", str(root)])


if __name__ == "__main__":
    raise SystemExit(run())
