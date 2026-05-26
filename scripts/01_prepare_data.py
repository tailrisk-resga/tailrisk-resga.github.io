from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.config import load_config
from utils.paths import ProjectPaths
from data.clean import (
    build_cross_sectional_samples,
    build_pointwise_samples,
    build_resga_samples,
    build_temporal_samples,
    clean_country_data,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ES prediction data from raw inputs.")
    parser.add_argument("--config", required=True, help="Path to a YAML pipeline config.")
    parser.add_argument(
        "--steps",
        nargs="+",
        default=None,
        choices=["clean", "pointwise", "temporal", "cross_sectional", "resga"],
        help="Subset of preparation steps to run. Defaults to config data.steps.",
    )
    parser.add_argument("--country", action="append", help="Run only selected country. Can be repeated.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = ProjectPaths.from_args(
        root=config.get("project", {}).get("root"),
        data_dir=config.get("project", {}).get("data_dir", "data"),
        prediction_root=config.get("project", {}).get("prediction_root", "outputs/predictions"),
        output_dir=config.get("project", {}).get("output_dir", "outputs"),
    )
    data_cfg = config.get("data", {})
    countries = args.country or config.get("countries", ["USA"])
    steps = args.steps or data_cfg.get("steps", ["clean", "pointwise", "temporal", "cross_sectional", "resga"])
    raw_subdir = data_cfg.get("raw_subdir", "raw/developed")
    processed_subdir = data_cfg.get("processed_subdir", "processed")
    sample_subdir = data_cfg.get("sample_subdir", "samples")
    sequence_length = int(data_cfg.get("sequence_length", config.get("training", {}).get("sequence_length", 12)))
    n_jobs = int(data_cfg.get("n_jobs", -1))
    sample_start = data_cfg.get("sample_start")
    sample_end = data_cfg.get("sample_end")

    for country in countries:
        print(f"== Preparing {country} ==")
        raw_path = paths.data_dir / raw_subdir / f"{country}.csv"
        processed_path = paths.data_dir / processed_subdir / f"{country}.csv"
        sample_root = paths.data_dir / sample_subdir / country
        temporal_dir = sample_root / "temporal"
        pointwise_dir = sample_root / "pointwise"
        cross_dir = sample_root / "cross_sectional"
        resga_dir = sample_root / "resga"

        if "clean" in steps:
            print(f"[clean] {raw_path} -> {processed_path}")
            clean_country_data(raw_path, processed_path, n_jobs=n_jobs)

        if "pointwise" in steps:
            print(f"[pointwise] {processed_path} -> {pointwise_dir}")
            build_pointwise_samples(
                processed_path,
                pointwise_dir,
                country=country,
                start=sample_start,
                end=sample_end,
            )

        if "temporal" in steps:
            print(f"[temporal] {processed_path} -> {temporal_dir}")
            build_temporal_samples(
                processed_path,
                temporal_dir,
                country=country,
                sequence_length=sequence_length,
                start=sample_start,
                end=sample_end,
            )

        if "cross_sectional" in steps:
            print(f"[cross_sectional] {temporal_dir} -> {cross_dir}")
            build_cross_sectional_samples(
                processed_path,
                temporal_dir,
                cross_dir,
                country=country,
                start=sample_start,
                end=sample_end,
            )

        if "resga" in steps:
            print(f"[resga] {processed_path} + {temporal_dir} -> {resga_dir}")
            build_resga_samples(
                processed_path,
                temporal_dir,
                resga_dir,
                country=country,
                sequence_length=sequence_length,
                start=data_cfg.get("resga_start"),
                end=sample_end,
            )

    print("Data preparation complete.")


if __name__ == "__main__":
    main()
