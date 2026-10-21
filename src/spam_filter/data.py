from __future__ import annotations

import hashlib
import io
import json
import re
import urllib.error
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from spam_filter.config import ProjectPaths

UCI_DATASET_ID = 228
DATASET_NAME = "SMS Spam Collection"
DATASET_DOI = "10.24432/C5CC84"
DATASET_LICENSE = "CC BY 4.0"
DEFAULT_SEED = 42
UCI_ARCHIVE_URL = "https://archive.ics.uci.edu/static/public/228/sms%2Bspam%2Bcollection.zip"
UCI_ARCHIVE_MEMBER = "SMSSpamCollection"


@dataclass(frozen=True, slots=True)
class SplitConfig:
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = DEFAULT_SEED

    def validate(self) -> None:
        total = self.train_fraction + self.validation_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Split fractions must sum to 1.0, got {total:.6f}")
        if min(self.train_fraction, self.validation_fraction, self.test_fraction) <= 0:
            raise ValueError("All split fractions must be positive.")


@dataclass(frozen=True, slots=True)
class PreparedDataPaths:
    raw_snapshot: Path
    canonical: Path
    train: Path
    validation: Path
    test: Path
    manifest: Path
    eda_json: Path
    eda_markdown: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_label(value: Any) -> tuple[str, int]:
    label_name = str(value).strip().lower()
    if label_name == "ham":
        return label_name, 0
    if label_name == "spam":
        return label_name, 1
    raise ValueError(f"Unexpected SMS label: {value!r}")


def canonicalize_source_frame(source: pd.DataFrame) -> pd.DataFrame:
    """Validate and canonicalize a raw two-column text/label dataframe."""
    required = {"text", "label_name"}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"Source frame is missing columns: {sorted(missing)}")

    frame = source.loc[:, ["text", "label_name"]].copy()
    if frame["text"].isna().any() or frame["label_name"].isna().any():
        raise ValueError("Dataset contains missing text or label values.")

    frame["text"] = frame["text"].astype(str)
    if frame["text"].str.strip().str.len().eq(0).any():
        raise ValueError("Dataset contains empty messages.")

    normalized = frame["label_name"].map(_normalize_label)
    frame["label_name"] = normalized.map(lambda pair: pair[0])
    frame["label"] = normalized.map(lambda pair: pair[1]).astype("int8")
    frame.insert(0, "source_row_id", range(len(frame)))
    frame["text_sha256"] = frame["text"].map(text_sha256)
    return frame


def _frame_from_uci(dataset: Any) -> pd.DataFrame:
    features = dataset.data.features
    targets = dataset.data.targets
    if features is None or targets is None:
        raise ValueError("UCI response did not contain features and targets.")
    if features.shape[1] < 1 or targets.shape[1] < 1:
        raise ValueError("UCI response has an unexpected empty schema.")

    return pd.DataFrame(
        {
            "text": features.iloc[:, 0],
            "label_name": targets.iloc[:, 0],
        }
    )


def _frame_from_uci_archive_bytes(payload: bytes) -> pd.DataFrame:
    """Parse the official UCI ZIP payload into the project's source schema."""
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = [name for name in archive.namelist() if Path(name).name == UCI_ARCHIVE_MEMBER]
            if not members:
                raise ValueError(
                    f"UCI archive does not contain expected member {UCI_ARCHIVE_MEMBER!r}."
                )
            raw_bytes = archive.read(members[0])
    except zipfile.BadZipFile as exc:
        raise ValueError("UCI download is not a valid ZIP archive.") from exc

    text = raw_bytes.decode("utf-8", errors="strict")
    rows: list[tuple[str, str]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            continue
        try:
            label_name, message = line.split("\t", maxsplit=1)
        except ValueError as exc:
            raise ValueError(
                f"Malformed UCI row {line_number}: expected <label>\t<message>."
            ) from exc
        rows.append((message, label_name))

    if not rows:
        raise ValueError("UCI archive contained no SMS rows.")
    return pd.DataFrame(rows, columns=["text", "label_name"])


def _fetch_uci_via_archive(timeout_seconds: float = 30.0) -> pd.DataFrame:
    request = urllib.request.Request(
        UCI_ARCHIVE_URL,
        headers={"User-Agent": "spam-filter-lab/0.3.0 (+UCI SMS Spam Collection)"},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        payload = response.read()
    return _frame_from_uci_archive_bytes(payload)


def _fetch_uci_via_ucimlrepo() -> pd.DataFrame:
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:  # pragma: no cover - guarded by project extras
        raise RuntimeError('Install data dependencies with: pip install -e ".[data]"') from exc

    dataset = fetch_ucirepo(id=UCI_DATASET_ID)
    return _frame_from_uci(dataset)


def fetch_uci_sms_frame() -> pd.DataFrame:
    """Fetch UCI SMS Spam Collection, preferring the official static archive.

    UCI currently exposes dataset 228 on its website and as a downloadable ZIP,
    while some ucimlrepo releases may reject the same dataset as unavailable for
    Python import. The static official archive is therefore the primary transport;
    ucimlrepo remains a compatibility fallback.
    """
    failures: list[str] = []

    try:
        return _fetch_uci_via_archive()
    except (
        OSError,
        TimeoutError,
        UnicodeDecodeError,
        ValueError,
        urllib.error.HTTPError,
        urllib.error.URLError,
    ) as exc:
        failures.append(f"official archive: {type(exc).__name__}: {exc}")

    try:
        return _fetch_uci_via_ucimlrepo()
    except Exception as exc:  # ucimlrepo uses package-specific exception classes
        failures.append(f"ucimlrepo fallback: {type(exc).__name__}: {exc}")

    detail = " | ".join(failures)
    raise RuntimeError(f"Unable to fetch {DATASET_NAME} from UCI. {detail}")


def _detect_conflicting_duplicates(frame: pd.DataFrame) -> list[str]:
    label_counts = frame.groupby("text", sort=False)["label"].nunique()
    return label_counts[label_counts > 1].index.astype(str).tolist()


def _split_frame(frame: pd.DataFrame, config: SplitConfig) -> dict[str, pd.DataFrame]:
    config.validate()
    train, holdout = train_test_split(
        frame,
        train_size=config.train_fraction,
        random_state=config.seed,
        stratify=frame["label"],
    )
    validation_share_of_holdout = config.validation_fraction / (
        config.validation_fraction + config.test_fraction
    )
    validation, test = train_test_split(
        holdout,
        train_size=validation_share_of_holdout,
        random_state=config.seed,
        stratify=holdout["label"],
    )
    return {
        "train": train.sort_values("source_row_id").reset_index(drop=True),
        "validation": validation.sort_values("source_row_id").reset_index(drop=True),
        "test": test.sort_values("source_row_id").reset_index(drop=True),
    }


def _split_overlap(splits: dict[str, pd.DataFrame]) -> dict[str, int]:
    hashes = {name: set(split["text_sha256"]) for name, split in splits.items()}
    return {
        "train_validation": len(hashes["train"] & hashes["validation"]),
        "train_test": len(hashes["train"] & hashes["test"]),
        "validation_test": len(hashes["validation"] & hashes["test"]),
    }


def _eda_summary(
    raw: pd.DataFrame,
    unique: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    duplicate_count: int,
) -> dict[str, Any]:
    lengths = unique["text"].str.len()
    url_mask = unique["text"].str.contains(r"(?:https?://|www\.)", flags=re.IGNORECASE, regex=True)
    digit_mask = unique["text"].str.contains(r"\d", regex=True)

    class_counts = unique["label_name"].value_counts().sort_index()
    split_stats: dict[str, Any] = {}
    for name, split in splits.items():
        counts = split["label_name"].value_counts().sort_index()
        split_stats[name] = {
            "rows": int(len(split)),
            "ham": int(counts.get("ham", 0)),
            "spam": int(counts.get("spam", 0)),
            "spam_rate": float(split["label"].mean()),
        }

    return {
        "dataset": DATASET_NAME,
        "uci_dataset_id": UCI_DATASET_ID,
        "rows_raw": int(len(raw)),
        "rows_unique": int(len(unique)),
        "exact_duplicate_texts_removed": int(duplicate_count),
        "class_counts": {key: int(value) for key, value in class_counts.items()},
        "spam_rate": float(unique["label"].mean()),
        "text_length_chars": {
            "min": int(lengths.min()),
            "median": float(lengths.median()),
            "mean": float(lengths.mean()),
            "p95": float(lengths.quantile(0.95)),
            "max": int(lengths.max()),
        },
        "url_message_rate": float(url_mask.mean()),
        "digit_message_rate": float(digit_mask.mean()),
        "splits": split_stats,
        "split_text_hash_overlap": _split_overlap(splits),
    }


def _write_eda_markdown(summary: dict[str, Any], path: Path) -> None:
    lengths = summary["text_length_chars"]
    splits = summary["splits"]
    lines = [
        "# M1 Data & EDA Report",
        "",
        f"- Dataset: {summary['dataset']} (UCI id {summary['uci_dataset_id']})",
        f"- Raw rows: {summary['rows_raw']}",
        f"- Unique messages: {summary['rows_unique']}",
        f"- Exact duplicate texts removed before split: {summary['exact_duplicate_texts_removed']}",
        f"- Spam rate after deduplication: {summary['spam_rate']:.4f}",
        f"- Median text length: {lengths['median']:.1f} chars",
        f"- 95th percentile text length: {lengths['p95']:.1f} chars",
        f"- Messages containing a URL-like token: {summary['url_message_rate']:.4f}",
        f"- Messages containing digits: {summary['digit_message_rate']:.4f}",
        "",
        "## Splits",
        "",
        "| split | rows | ham | spam | spam rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("train", "validation", "test"):
        item = splits[name]
        lines.append(
            f"| {name} | {item['rows']} | {item['ham']} | {item['spam']} | "
            f"{item['spam_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Leakage checks",
            "",
            "All split-pair text SHA-256 overlap counts must be zero.",
            "",
            "```json",
            json.dumps(summary["split_text_hash_overlap"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def prepare_from_frame(
    source: pd.DataFrame,
    paths: ProjectPaths,
    config: SplitConfig | None = None,
) -> PreparedDataPaths:
    """Create the complete M1 raw/processed/split/report contract from a source frame."""
    config = config or SplitConfig()
    config.validate()
    paths.ensure_runtime_dirs()

    raw = canonicalize_source_frame(source)
    conflicting = _detect_conflicting_duplicates(raw)
    if conflicting:
        raise ValueError(
            f"Found {len(conflicting)} message texts with conflicting labels; refusing to split."
        )

    duplicate_count = int(raw.duplicated(subset=["text"], keep="first").sum())
    unique = raw.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    splits = _split_frame(unique, config)
    overlap = _split_overlap(splits)
    if any(overlap.values()):
        raise AssertionError(f"Text leakage detected across splits: {overlap}")

    raw_path = paths.data_raw / "sms_spam_collection.csv"
    canonical_path = paths.data_processed / "sms_spam_unique.csv"
    train_path = paths.data_processed / "train.csv"
    validation_path = paths.data_processed / "validation.csv"
    test_path = paths.data_processed / "test.csv"
    manifest_path = paths.data_processed / "m1_manifest.json"
    eda_json_path = paths.reports / "m1_eda.json"
    eda_markdown_path = paths.reports / "m1_eda.md"

    raw.to_csv(raw_path, index=False)
    unique.to_csv(canonical_path, index=False)
    splits["train"].to_csv(train_path, index=False)
    splits["validation"].to_csv(validation_path, index=False)
    splits["test"].to_csv(test_path, index=False)

    eda = _eda_summary(raw, unique, splits, duplicate_count)
    eda_json_path.write_text(json.dumps(eda, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_eda_markdown(eda, eda_markdown_path)

    files = {
        "raw_snapshot": raw_path,
        "canonical": canonical_path,
        "train": train_path,
        "validation": validation_path,
        "test": test_path,
        "eda_json": eda_json_path,
        "eda_markdown": eda_markdown_path,
    }
    manifest = {
        "milestone": "M1",
        "dataset": {
            "name": DATASET_NAME,
            "uci_dataset_id": UCI_DATASET_ID,
            "doi": DATASET_DOI,
            "license": DATASET_LICENSE,
        },
        "split_config": asdict(config),
        "rows_raw": len(raw),
        "rows_unique": len(unique),
        "exact_duplicate_texts_removed": duplicate_count,
        "split_text_hash_overlap": overlap,
        "files": {
            name: {
                "path": str(path.relative_to(paths.root)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for name, path in files.items()
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return PreparedDataPaths(
        raw_snapshot=raw_path,
        canonical=canonical_path,
        train=train_path,
        validation=validation_path,
        test=test_path,
        manifest=manifest_path,
        eda_json=eda_json_path,
        eda_markdown=eda_markdown_path,
    )


def prepare_uci_dataset(
    paths: ProjectPaths,
    config: SplitConfig | None = None,
) -> PreparedDataPaths:
    return prepare_from_frame(fetch_uci_sms_frame(), paths=paths, config=config)


def load_manifest(paths: ProjectPaths) -> dict[str, Any]:
    manifest_path = paths.data_processed / "m1_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"M1 manifest not found at {manifest_path}. Run `make m1-data` first."
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))
