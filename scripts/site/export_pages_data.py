from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PUBLIC_COLUMNS = ["ticker", "name", "eom", "v", "e"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export public prediction files for GitHub Pages.")
    parser.add_argument("--country", default="USA")
    parser.add_argument("--prediction-root", default="outputs/site_predictions")
    parser.add_argument("--output-root", default="docs/data/predictions")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = Path(args.prediction_root) / args.country
    output_dir = Path(args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    dates = []
    for csv_path in sorted(source_dir.glob("*.csv")):
        if csv_path.stem == "latest" or csv_path.stem.endswith("_per_seed"):
            continue
        date = pd.Timestamp(csv_path.stem).strftime("%Y-%m-%d")
        _write_records(csv_path, output_dir / f"{date}.json")
        dates.append({"date": date, "file": f"{date}.json"})

    latest_path = source_dir / "latest.csv"
    if latest_path.exists():
        _write_records(latest_path, output_dir / "latest.json")

    dates = sorted(dates, key=lambda row: row["date"], reverse=True)
    (output_dir / "dates.json").write_text(json.dumps(dates, indent=2), encoding="utf-8")


def _write_records(csv_path: Path, json_path: Path) -> None:
    df = pd.read_csv(csv_path)
    for col in PUBLIC_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    records = df[PUBLIC_COLUMNS].where(pd.notnull(df[PUBLIC_COLUMNS]), None).to_dict(orient="records")
    json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
