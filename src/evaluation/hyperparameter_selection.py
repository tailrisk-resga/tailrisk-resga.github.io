from __future__ import annotations

import pandas as pd


def summarize_validation_losses(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize validation losses by model and hyperparameter."""

    required = {"model", "hyperparameter", "loss"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Validation predictions are missing required columns: {sorted(missing)}")

    valid = predictions[predictions["split"] == "valid"].copy()
    if valid.empty:
        raise ValueError("No validation predictions found. Expected files named valid.csv.")

    return (
        valid.groupby(["model", "hyperparameter"], dropna=False)
        .agg(
            mean_valid_loss=("loss", "mean"),
            std_valid_loss=("loss", "std"),
            n_obs=("loss", "count"),
            n_seeds=("seed", "nunique"),
            n_windows=("window", "nunique"),
        )
        .reset_index()
        .sort_values(["model", "mean_valid_loss", "hyperparameter"])
        .reset_index(drop=True)
    )


def select_best_hyperparameters(validation_summary: pd.DataFrame) -> pd.DataFrame:
    """Select the lowest validation-loss hyperparameter for each model."""

    if validation_summary.empty:
        raise ValueError("Validation summary is empty.")
    idx = validation_summary.groupby("model")["mean_valid_loss"].idxmin()
    return (
        validation_summary.loc[idx]
        .sort_values("model")
        .reset_index(drop=True)
    )


def filter_predictions_to_best_hyperparameters(
    predictions: pd.DataFrame,
    best_hyperparameters: pd.DataFrame,
) -> pd.DataFrame:
    """Keep baseline predictions and neural runs selected by validation loss."""

    required_predictions = {"model", "hyperparameter"}
    missing_predictions = required_predictions - set(predictions.columns)
    if missing_predictions:
        raise ValueError(
            f"Predictions are missing required columns: {sorted(missing_predictions)}"
        )

    required_best = {"model", "hyperparameter"}
    missing_best = required_best - set(best_hyperparameters.columns)
    if missing_best:
        raise ValueError(
            f"Best-hyperparameter table is missing required columns: {sorted(missing_best)}"
        )

    best_pairs = set(
        zip(
            best_hyperparameters["model"].astype(str),
            best_hyperparameters["hyperparameter"].astype(str),
        )
    )
    model_hyperparameters = list(
        zip(predictions["model"].astype(str), predictions["hyperparameter"].astype(str))
    )

    keep_baseline = predictions["hyperparameter"].isna()
    keep_selected = pd.Series(
        [pair in best_pairs for pair in model_hyperparameters],
        index=predictions.index,
    )
    return predictions[keep_baseline | keep_selected].reset_index(drop=True)


__all__ = [
    "filter_predictions_to_best_hyperparameters",
    "select_best_hyperparameters",
    "summarize_validation_losses",
]
