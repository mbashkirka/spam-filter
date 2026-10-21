from __future__ import annotations

from pathlib import Path

from spam_filter.config import ProjectPaths
from spam_filter.data import SplitConfig, prepare_uci_dataset


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    prepared = prepare_uci_dataset(ProjectPaths.from_root(root), SplitConfig(seed=42))
    print(f"manifest={prepared.manifest.relative_to(root)}")
    print(f"eda={prepared.eda_markdown.relative_to(root)}")
    print("status=prepared")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
