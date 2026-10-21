from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from spam_filter.config import ProjectPaths
from spam_filter.data import load_manifest, sha256_file


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    paths = ProjectPaths.from_root(root)
    manifest = load_manifest(paths)

    for item in manifest["files"].values():
        path = root / item["path"]
        if not path.exists():
            raise SystemExit(f"missing={path}")
        if sha256_file(path) != item["sha256"]:
            raise SystemExit(f"hash_mismatch={path}")

    split_names = ("train", "validation", "test")
    splits = {
        name: pd.read_csv(paths.data_processed / f"{name}.csv") for name in split_names
    }
    hash_sets = {name: set(frame["text_sha256"]) for name, frame in splits.items()}
    overlaps = {
        "train_validation": len(hash_sets["train"] & hash_sets["validation"]),
        "train_test": len(hash_sets["train"] & hash_sets["test"]),
        "validation_test": len(hash_sets["validation"] & hash_sets["test"]),
    }
    if any(overlaps.values()):
        raise SystemExit(f"split_leakage={json.dumps(overlaps, sort_keys=True)}")

    print("dataset=uci-sms-spam-collection")
    print(f"rows_raw={manifest['rows_raw']}")
    print(f"rows_unique={manifest['rows_unique']}")
    print(f"duplicates_removed={manifest['exact_duplicate_texts_removed']}")
    print(f"train={len(splits['train'])}")
    print(f"validation={len(splits['validation'])}")
    print(f"test={len(splits['test'])}")
    print("split_overlap=0")
    print("milestone=M1")
    print("status=ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
