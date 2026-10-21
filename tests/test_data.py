import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from spam_filter.config import ProjectPaths
from spam_filter.data import (
    SplitConfig,
    _frame_from_uci_archive_bytes,
    canonicalize_source_frame,
    prepare_from_frame,
    sha256_file,
)


def _source_frame(n_per_class: int = 20) -> pd.DataFrame:
    ham = [f"normal message {i}" for i in range(n_per_class)]
    spam = [f"WIN prize now {i} call 0800" for i in range(n_per_class)]
    return pd.DataFrame(
        {
            "text": ham + spam + [ham[0], spam[0]],
            "label_name": ["ham"] * n_per_class + ["spam"] * n_per_class + ["ham", "spam"],
        }
    )


def test_canonicalize_labels_and_hashes() -> None:
    frame = canonicalize_source_frame(_source_frame(5))
    assert set(frame["label"].unique()) == {0, 1}
    assert frame["text_sha256"].str.len().eq(64).all()


def test_conflicting_duplicate_is_rejected(tmp_path: Path) -> None:
    source = pd.DataFrame(
        {"text": ["same message", "same message"], "label_name": ["ham", "spam"]}
    )
    with pytest.raises(ValueError, match="conflicting labels"):
        prepare_from_frame(source, ProjectPaths.from_root(tmp_path))


def test_prepare_data_contract_has_no_split_leakage(tmp_path: Path) -> None:
    source = _source_frame(20)
    paths = ProjectPaths.from_root(tmp_path)
    prepared = prepare_from_frame(source, paths, SplitConfig(seed=7))

    train = pd.read_csv(prepared.train)
    validation = pd.read_csv(prepared.validation)
    test = pd.read_csv(prepared.test)

    assert len(train) + len(validation) + len(test) == 40
    assert set(train["text_sha256"]).isdisjoint(validation["text_sha256"])
    assert set(train["text_sha256"]).isdisjoint(test["text_sha256"])
    assert set(validation["text_sha256"]).isdisjoint(test["text_sha256"])
    assert prepared.manifest.exists()
    assert prepared.eda_json.exists()
    assert prepared.eda_markdown.exists()
    assert len(sha256_file(prepared.train)) == 64


def test_parse_official_uci_archive_shape() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "SMSSpamCollection",
            "ham\tHello there\nspam\tWIN a prize now!\n",
        )

    frame = _frame_from_uci_archive_bytes(buffer.getvalue())
    assert frame.to_dict(orient="records") == [
        {"text": "Hello there", "label_name": "ham"},
        {"text": "WIN a prize now!", "label_name": "spam"},
    ]


def test_parse_official_uci_archive_rejects_wrong_member() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("unexpected.txt", "ham\tHello")

    with pytest.raises(ValueError, match="expected member"):
        _frame_from_uci_archive_bytes(buffer.getvalue())
