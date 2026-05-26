from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluation.hyperparameter_selection import (
    select_best_hyperparameters,
    summarize_validation_losses,
)
from evaluation.predictions import aggregate_prediction_files
from utils.paths import ProjectPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select model hyperparameters using validation predictions."
    )
    parser.add_argument("--prediction-root", default="outputs/predictions")
    parser.add_argument("--country", required=True)
    parser.add_argument("--output-dir", default="outputs/metrics")
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.from_args(
        prediction_root=args.prediction_root,
        output_dir=args.output_dir,
    )
    out_dir = paths.output_dir / args.country
    out_dir.mkdir(parents=True, exist_ok=True)

    validation_predictions = aggregate_prediction_files(
        prediction_root=paths.prediction_root,
        country=args.country,
        alpha=args.alpha,
        split="valid",
    )
    validation_summary = summarize_validation_losses(validation_predictions)
    best_hyperparameters = select_best_hyperparameters(validation_summary)

    predictions_path = out_dir / "validation_predictions_with_loss.csv"
    summary_path = out_dir / "validation_loss_summary.csv"
    best_path = out_dir / "best_hyperparameters.csv"

    validation_predictions.to_csv(predictions_path, index=False)
    validation_summary.to_csv(summary_path, index=False)
    best_hyperparameters.to_csv(best_path, index=False)

    print(f"Saved validation predictions to {predictions_path}")
    print(f"Saved validation loss summary to {summary_path}")
    print(f"Saved best hyperparameters to {best_path}")


if __name__ == "__main__":
    main()
