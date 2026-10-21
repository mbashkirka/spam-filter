from pathlib import Path

import pandas as pd

from spam_filter.config import ProjectPaths
from spam_filter.m2 import M2Config, binary_metrics, run_m2


def test_binary_metrics() -> None:
    metrics = binary_metrics([0, 0, 1, 1], [0.1, 0.8, 0.7, 0.9])
    assert metrics["fp"] == 1
    assert metrics["fn"] == 0
    assert 0.0 <= metrics["pr_auc"] <= 1.0


def test_m2_pipeline_uses_existing_splits(tmp_path: Path) -> None:
    paths = ProjectPaths.from_root(tmp_path)
    paths.ensure_runtime_dirs()
    rows = []
    for i in range(40):
        rows.append({"text": f"normal project meeting schedule {i}", "label": 0})
        rows.append({"text": f"win free cash prize call now {i}", "label": 1})
    frame = pd.DataFrame(rows)
    frame.iloc[:50].to_csv(paths.data_processed / "train.csv", index=False)
    frame.iloc[50:65].to_csv(paths.data_processed / "validation.csv", index=False)
    frame.iloc[65:].to_csv(paths.data_processed / "test.csv", index=False)

    result = run_m2(paths, M2Config(alpha=1.0, min_df=1))
    assert result["milestone"] == "M2"
    assert len(result["validation_results"]) == 4
    assert Path(result["model_path"]).exists()
    assert Path(result["report_markdown"]).exists()
