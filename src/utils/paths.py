from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Centralized filesystem layout for reproducible scripts."""

    root: Path
    data_dir: Path
    prediction_root: Path
    output_dir: Path

    @classmethod
    def from_args(
        cls,
        data_dir: str | os.PathLike[str] | None = None,
        prediction_root: str | os.PathLike[str] | None = None,
        output_dir: str | os.PathLike[str] | None = None,
        root: str | os.PathLike[str] | None = None,
    ) -> "ProjectPaths":
        project_root = Path(
            root or os.environ.get("ES_PROJECT_ROOT") or Path.cwd()
        ).resolve()
        return cls(
            root=project_root,
            data_dir=_resolve(project_root, data_dir or "data"),
            prediction_root=_resolve(project_root, prediction_root or "outputs/predictions"),
            output_dir=_resolve(project_root, output_dir or "outputs/metrics"),
        )


def _resolve(root: Path, value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()
