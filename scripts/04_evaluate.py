from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.paths import ProjectPaths
from evaluation.hyperparameter_selection import filter_predictions_to_best_hyperparameters
from evaluation.predictions import aggregate_prediction_files, summarize_losses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate expected-shortfall prediction CSV files."
    )
    parser.add_argument("--prediction-root", default="outputs/predictions")
    parser.add_argument("--country", required=True)
    parser.add_argument("--output-dir", default="outputs/metrics")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--split", default="test", choices=["valid", "test", "insample", "all"])
    parser.add_argument(
        "--best-hyperparameters",
        default=None,
        help=(
            "Path to best_hyperparameters.csv. Defaults to "
            "<output-dir>/<country>/best_hyperparameters.csv."
        ),
    )
    parser.add_argument(
        "--all-hyperparameters",
        action="store_true",
        help="Evaluate every hyperparameter instead of filtering to validation-selected runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.from_args(
        prediction_root=args.prediction_root,
        output_dir=args.output_dir,
    )
    out_dir = paths.output_dir / args.country
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions = aggregate_prediction_files(
        prediction_root=paths.prediction_root,
        country=args.country,
        alpha=args.alpha,
        split=None if args.split == "all" else args.split,
    )

    if not args.all_hyperparameters:
        best_path = (
            Path(args.best_hyperparameters).resolve()
            if args.best_hyperparameters is not None
            else out_dir / "best_hyperparameters.csv"
        )
        if not best_path.exists():
            raise FileNotFoundError(
                f"Best-hyperparameter file not found: {best_path}. "
                "Run scripts/03_tune_params.py first, or pass --all-hyperparameters."
            )
        predictions = filter_predictions_to_best_hyperparameters(
            predictions,
            best_hyperparameters=pd.read_csv(best_path),
        )
        if predictions.empty:
            raise ValueError(
                f"No predictions match the selected hyperparameters in {best_path}."
            )

    summary = summarize_losses(predictions)

    selection_name = "all_hyperparameters" if args.all_hyperparameters else "selected_hyperparameters"
    output_prefix = f"{args.split}_{selection_name}"
    predictions_path = Path(out_dir) / f"{output_prefix}_predictions_with_loss.csv"
    summary_path = Path(out_dir) / f"{output_prefix}_loss_summary.csv"
    predictions.to_csv(predictions_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"Saved aggregated predictions to {predictions_path}")
    print(f"Saved loss summary to {summary_path}")


if __name__ == "__main__":
    main()
