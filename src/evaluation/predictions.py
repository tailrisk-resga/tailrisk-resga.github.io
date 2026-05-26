from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from evaluation.losses import add_fz0_loss

REQUIRED_PREDICTION_COLUMNS = ("id", "eom", "v", "e", "y")
SPLIT_FILE_NAMES = {"valid", "test", "insample"}


def validate_prediction_frame(
    df: pd.DataFrame,
    source: str | Path = "<dataframe>",
    required_columns: Iterable[str] = REQUIRED_PREDICTION_COLUMNS,
) -> None:
    """Validate the public prediction CSV schema."""

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing required prediction columns: {missing}")


def read_prediction_file(
    path: str | Path,
    model: str | None = None,
    run: str | None = None,
    hyperparameter: str | None = None,
    seed: str | int | None = None,
    window: str | None = None,
    split: str | None = None,
) -> pd.DataFrame:
    """Read one prediction CSV and attach model/run metadata when absent."""

    csv_path = Path(path)
    df = pd.read_csv(csv_path)
    validate_prediction_frame(df, source=csv_path)
    if "model" not in df.columns and model is not None:
        df["model"] = model
    if "run" not in df.columns and run is not None:
        df["run"] = run
    if "hyperparameter" not in df.columns and hyperparameter is not None:
        df["hyperparameter"] = hyperparameter
    if "seed" not in df.columns and seed is not None:
        df["seed"] = str(seed).replace("seed_", "")
    if "window" not in df.columns and window is not None:
        df["window"] = window
    if "split" not in df.columns and split is not None:
        df["split"] = split
    df["source_file"] = str(csv_path)
    return df


def _prediction_metadata(path: Path, country_root: Path) -> dict:
    rel = path.relative_to(country_root)
    parts = rel.parts
    model = parts[0] if len(parts) >= 1 else None

    if len(parts) >= 5:
        hyperparameter = parts[1]
        seed = parts[2]
        window = parts[3]
    else:
        hyperparameter = None
        seed = None
        window = None

    # Single-file baselines such as GAS/GARCH do not have validation outputs.
    # Their files are therefore treated as test predictions by convention.
    split = path.stem if path.stem in SPLIT_FILE_NAMES else "test"
    run = "/".join(part for part in [hyperparameter, seed, window] if part) or path.stem
    return {
        "model": model,
        "run": run,
        "hyperparameter": hyperparameter,
        "seed": seed,
        "window": window,
        "split": split,
    }


def discover_prediction_files(
    prediction_root: str | Path,
    country: str,
    split: str | None = None,
) -> list[Path]:
    """Find prediction CSV files for one country under a model/run hierarchy."""

    country_root = Path(prediction_root) / country
    if not country_root.exists():
        raise FileNotFoundError(f"Prediction country directory not found: {country_root}")

    paths = []
    for path in country_root.rglob("*.csv"):
        if not path.is_file():
            continue
        metadata = _prediction_metadata(path, country_root)
        if split is None or metadata["split"] == split:
            paths.append(path)
    return sorted(paths)


def aggregate_prediction_files(
    prediction_root: str | Path,
    country: str,
    alpha: float = 0.05,
    split: str | None = None,
) -> pd.DataFrame:
    """Aggregate all prediction CSVs for ``country`` and recompute FZ0 loss."""

    frames = []
    country_root = Path(prediction_root) / country
    for path in discover_prediction_files(prediction_root, country, split=split):
        metadata = _prediction_metadata(path, country_root)
        frames.append(
            read_prediction_file(
                path,
                **metadata,
            )
        )

    if not frames:
        raise FileNotFoundError(f"No prediction CSV files found under: {country_root}")

    out = pd.concat(frames, ignore_index=True)
    out["eom"] = pd.to_datetime(out["eom"]).dt.strftime("%Y-%m-%d")
    out = add_fz0_loss(out, alpha=alpha)
    sort_cols = [
        col for col in ["split", "model", "hyperparameter", "seed", "window", "run", "eom", "id"]
        if col in out.columns
    ]
    return out.sort_values(sort_cols).reset_index(drop=True)


def summarize_losses(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize FZ0 losses by model and run/seed when available."""

    group_cols = ["model"]
    for col in ["split", "hyperparameter", "seed", "window", "run"]:
        if col in predictions.columns:
            group_cols.append(col)

    return (
        predictions.groupby(group_cols, dropna=False)["loss"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_fz0_loss", "std": "std_fz0_loss"})
    )
