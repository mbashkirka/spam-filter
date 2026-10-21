from __future__ import annotations

from pathlib import Path

from spam_filter.config import ProjectPaths
from spam_filter.m6 import prepare_enron1_dataset


def main() -> int:
    paths = ProjectPaths.from_root(Path.cwd())
    result = prepare_enron1_dataset(paths)
    print(f"dataset={result.csv_path.relative_to(paths.root)}")
    print(f"manifest={result.manifest_path.relative_to(paths.root)}")
    print(f"rows_raw={result.rows_raw}")
    print(f"rows_unique={result.rows_unique}")
    print(f"duplicates_removed={result.duplicates_removed}")
    print("milestone=M6")
    print("status=prepared")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
