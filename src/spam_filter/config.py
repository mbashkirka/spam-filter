from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """Resolved repository paths used by milestone scripts."""

    root: Path
    data_raw: Path
    data_processed: Path
    artifacts: Path
    reports: Path

    @classmethod
    def from_root(cls, root: str | Path) -> ProjectPaths:
        resolved = Path(root).expanduser().resolve()
        return cls(
            root=resolved,
            data_raw=resolved / "data" / "raw",
            data_processed=resolved / "data" / "processed",
            artifacts=resolved / "artifacts",
            reports=resolved / "reports",
        )

    def ensure_runtime_dirs(self) -> None:
        for path in (self.data_raw, self.data_processed, self.artifacts, self.reports):
            path.mkdir(parents=True, exist_ok=True)
