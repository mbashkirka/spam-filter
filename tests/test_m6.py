from __future__ import annotations

import io
import ssl
import tarfile
from pathlib import Path

import numpy as np
import pytest

from spam_filter.m6 import (
    M6Config,
    _binary_metrics,
    _validate_downloaded_archive,
    _verified_ssl_context,
    parse_enron1_archive,
)


def _write_tar(path: Path, entries: dict[str, str]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name, text in entries.items():
            payload = text.encode("utf-8")
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def test_m6_config_contract() -> None:
    M6Config().validate()
    with pytest.raises(ValueError):
        M6Config(batch_size=0).validate()
    with pytest.raises(ValueError):
        M6Config(device="tpu").validate()


def test_parse_enron_archive(tmp_path: Path) -> None:
    archive = tmp_path / "enron1.tar.gz"
    _write_tar(
        archive,
        {
            "enron1/ham/0001.ham.txt": "meeting tomorrow at noon",
            "enron1/spam/0001.spam.txt": "claim your prize now",
            "enron1/Summary.txt": "ignored",
        },
    )
    frame = parse_enron1_archive(archive)
    assert len(frame) == 2
    assert frame["label"].tolist() == [0, 1]
    assert set(frame["label_name"]) == {"ham", "spam"}
    assert frame["text_sha256"].nunique() == 2


def test_parse_enron_detects_label_conflict(tmp_path: Path) -> None:
    archive = tmp_path / "enron1.tar.gz"
    _write_tar(
        archive,
        {
            "enron1/ham/a.txt": "same text",
            "enron1/spam/b.txt": "same text",
        },
    )
    with pytest.raises(ValueError, match="label conflicts"):
        parse_enron1_archive(archive)


def test_binary_metrics() -> None:
    y = np.asarray([0, 0, 1, 1], dtype=np.int8)
    pred = np.asarray([0, 1, 1, 1], dtype=np.int8)
    scores = np.asarray([0.1, 0.6, 0.8, 0.9])
    metrics = _binary_metrics(y, pred, scores)
    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 0
    assert metrics["tp"] == 2
    assert 0.0 <= metrics["pr_auc"] <= 1.0


def test_verified_ssl_context_requires_certificates() -> None:
    context = _verified_ssl_context()
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True


def test_validate_downloaded_archive(tmp_path: Path) -> None:
    archive = tmp_path / "enron1.tar.gz.part"
    _write_tar(
        archive,
        {
            "enron1/ham/0001.ham.txt": "h" * 6000,
            "enron1/spam/0001.spam.txt": "s" * 6000,
        },
    )
    # Highly repetitive gzip data can be <10 KiB, so pad with valid gzip trailing bytes.
    with archive.open("ab") as handle:
        handle.write(b"\0" * 12_000)
    _validate_downloaded_archive(archive)


def test_validate_downloaded_archive_rejects_non_tar(tmp_path: Path) -> None:
    archive = tmp_path / "bad.tar.gz.part"
    archive.write_bytes(b"x" * 12_000)
    with pytest.raises(RuntimeError, match="valid gzip tar archive"):
        _validate_downloaded_archive(archive)
