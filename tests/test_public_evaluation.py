from pathlib import Path

import pandas as pd

from evaluation.hyperparameter_selection import filter_predictions_to_best_hyperparameters
from evaluation.predictions import aggregate_prediction_files, summarize_losses


def test_aggregate_prediction_files_from_public_schema(tmp_path: Path) -> None:
    pred_dir = tmp_path / "predictions" / "USA" / "ReSGA" / "1_16_10" / "seed_1" / "2020-01-01_2021-01-01"
    pred_dir.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "id": [1, 2],
            "eom": ["2020-01-31", "2020-01-31"],
            "v": [-0.05, -0.04],
            "e": [-0.08, -0.07],
            "y": [-0.06, -0.02],
        }
    )
    frame.to_csv(pred_dir / "test.csv", index=False)
    frame.to_csv(pred_dir / "valid.csv", index=False)

    garch_dir = tmp_path / "predictions" / "USA" / "GARCH"
    garch_dir.mkdir(parents=True)
    frame.to_csv(garch_dir / "GARCH.csv", index=False)

    aggregated = aggregate_prediction_files(tmp_path / "predictions", country="USA")
    assert set(["model", "hyperparameter", "run", "seed", "window", "split", "loss"]).issubset(aggregated.columns)
    assert sorted(aggregated["model"].unique().tolist()) == ["GARCH", "ReSGA"]
    assert "1_16_10" in aggregated["hyperparameter"].unique().tolist()
    assert "1" in aggregated["seed"].unique().tolist()
    assert sorted(aggregated["split"].unique().tolist()) == ["test", "valid"]

    valid = aggregate_prediction_files(tmp_path / "predictions", country="USA", split="valid")
    assert valid["model"].unique().tolist() == ["ReSGA"]
    assert valid["split"].unique().tolist() == ["valid"]

    test = aggregate_prediction_files(tmp_path / "predictions", country="USA", split="test")
    assert sorted(test["model"].unique().tolist()) == ["GARCH", "ReSGA"]
    assert test["split"].unique().tolist() == ["test"]

    summary = summarize_losses(test)
    assert sorted(summary["model"].unique().tolist()) == ["GARCH", "ReSGA"]


def test_filter_predictions_to_best_hyperparameters_keeps_baselines() -> None:
    predictions = pd.DataFrame(
        {
            "model": ["ReSGA", "ReSGA", "GARCH"],
            "hyperparameter": ["1_16_10", "1_32_10", None],
            "loss": [1.0, 2.0, 3.0],
        }
    )
    best = pd.DataFrame({"model": ["ReSGA"], "hyperparameter": ["1_16_10"]})

    filtered = filter_predictions_to_best_hyperparameters(predictions, best)

    assert filtered["model"].tolist() == ["ReSGA", "GARCH"]
    assert filtered["hyperparameter"].tolist() == ["1_16_10", None]
