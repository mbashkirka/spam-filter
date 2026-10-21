from pathlib import Path

from spam_filter.config import ProjectPaths


def test_paths_resolve_from_root(tmp_path: Path) -> None:
    paths = ProjectPaths.from_root(tmp_path)
    assert paths.root == tmp_path.resolve()
    assert paths.data_raw == tmp_path.resolve() / "data" / "raw"
    assert paths.artifacts == tmp_path.resolve() / "artifacts"


def test_ensure_runtime_dirs(tmp_path: Path) -> None:
    paths = ProjectPaths.from_root(tmp_path)
    paths.ensure_runtime_dirs()
    assert paths.data_raw.is_dir()
    assert paths.data_processed.is_dir()
    assert paths.artifacts.is_dir()
    assert paths.reports.is_dir()
