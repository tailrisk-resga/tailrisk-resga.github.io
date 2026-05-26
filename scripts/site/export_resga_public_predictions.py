from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PUBLIC_COLUMNS = ["id", "ticker", "name", "eom", "target_month", "y", "v", "e", "n_seeds"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Average ReSGA seed predictions and export public site data.")
    parser.add_argument(
        "--input-root",
        default="outputs/source_results/USA/Retrieval",
        help="Root containing seed_<seed>/<hyperparameter>/<window>/test.csv.",
    )
    parser.add_argument("--hyperparameter", default="1_512_10")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument(
        "--output-root",
        default="outputs/public_site_data/USA/ReSGA/1_512_10",
        help="Directory where averaged CSV/JSON files will be written.",
    )
    parser.add_argument("--github-pages-root", default=None, help="Optional docs/data/predictions directory to update.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    averaged = _load_and_average(input_root, args.hyperparameter, args.seeds)
    averaged = averaged.sort_values(["eom", "id"]).reset_index(drop=True)

    _write_archive(averaged, output_root)
    if args.github_pages_root:
        _write_pages_data(averaged, Path(args.github_pages_root))


def _load_and_average(input_root: Path, hyperparameter: str, seeds: list[int]) -> pd.DataFrame:
    frames = []
    expected_windows: set[str] | None = None

    for seed in seeds:
        seed_dir = input_root / f"seed_{seed}" / hyperparameter
        paths = sorted(seed_dir.glob("*/test.csv"))
        if not paths:
            raise FileNotFoundError(f"No test.csv files found under {seed_dir}")

        windows = {path.parent.name for path in paths}
        if expected_windows is None:
            expected_windows = windows
        elif windows != expected_windows:
            missing = sorted(expected_windows - windows)
            extra = sorted(windows - expected_windows)
            raise ValueError(f"seed_{seed} windows mismatch; missing={missing}, extra={extra}")

        for path in paths:
            df = pd.read_csv(path)
            required = {"id", "eom", "y", "v", "e"}
            missing = sorted(required - set(df.columns))
            if missing:
                raise ValueError(f"{path} is missing required columns: {missing}")
            df = df[["id", "eom", "y", "v", "e"]].copy()
            df["seed"] = seed
            df["window"] = path.parent.name
            frames.append(df)

    all_rows = pd.concat(frames, ignore_index=True)
    all_rows["eom"] = pd.to_datetime(all_rows["eom"]).dt.strftime("%Y-%m-%d")
    grouped = (
        all_rows.groupby(["id", "eom"], as_index=False)
        .agg(
            y=("y", "mean"),
            v=("v", "mean"),
            e=("e", "mean"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(["eom", "id"])
    )

    seed_count = len(seeds)
    incomplete = grouped[grouped["n_seeds"] != seed_count]
    if not incomplete.empty:
        raise ValueError(f"{len(incomplete)} id/eom rows do not have all {seed_count} seeds.")

    eom = pd.to_datetime(grouped["eom"])
    grouped["target_month"] = (eom + pd.offsets.MonthEnd(1)).dt.strftime("%Y-%m-%d")
    grouped["ticker"] = ""
    grouped["name"] = ""
    return grouped[PUBLIC_COLUMNS]


def _write_archive(df: pd.DataFrame, output_root: Path) -> None:
    monthly_dir = output_root / "monthly"
    annual_dir = output_root / "annual"
    monthly_dir.mkdir(parents=True, exist_ok=True)
    annual_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_root / "all_predictions.csv", index=False)
    _write_json(output_root / "all_predictions.json", df)

    dates = []
    for eom, month_df in df.groupby("eom", sort=True):
        token = pd.Timestamp(eom).strftime("%Y-%m")
        month_df = month_df.sort_values("id")
        month_df.to_csv(monthly_dir / f"{token}.csv", index=False)
        _write_json(monthly_dir / f"{token}.json", month_df)
        dates.append({"date": eom, "month": token, "file": f"monthly/{token}.json"})

    for year, year_df in df.groupby(pd.to_datetime(df["eom"]).dt.year, sort=True):
        year_df = year_df.sort_values(["eom", "id"])
        year_df.to_csv(annual_dir / f"{year}.csv", index=False)
        _write_json(annual_dir / f"{year}.json", year_df)

    dates = sorted(dates, key=lambda row: row["date"], reverse=True)
    (output_root / "dates.json").write_text(json.dumps(dates, indent=2), encoding="utf-8")
    if dates:
        latest_df = df[df["eom"] == dates[0]["date"]].sort_values("id")
        latest_df.to_csv(output_root / "latest.csv", index=False)
        _write_json(output_root / "latest.json", latest_df)


def _write_pages_data(df: pd.DataFrame, pages_root: Path) -> None:
    pages_root.mkdir(parents=True, exist_ok=True)
    monthly_dir = pages_root / "monthly"
    annual_dir = pages_root / "annual"
    stocks_dir = pages_root / "stocks"
    monthly_dir.mkdir(parents=True, exist_ok=True)
    annual_dir.mkdir(parents=True, exist_ok=True)
    stocks_dir.mkdir(parents=True, exist_ok=True)

    dates = []
    for eom, month_df in df.groupby("eom", sort=True):
        token = pd.Timestamp(eom).strftime("%Y-%m")
        month_df = month_df.sort_values("id")
        _write_json(monthly_dir / f"{token}.json", month_df)
        dates.append({"date": eom, "month": token, "file": f"monthly/{token}.json"})

    for year, year_df in df.groupby(pd.to_datetime(df["eom"]).dt.year, sort=True):
        year_df = year_df.sort_values(["eom", "id"])
        _write_json(annual_dir / f"{year}.json", year_df)

    dates = sorted(dates, key=lambda row: row["date"], reverse=True)
    (pages_root / "dates.json").write_text(json.dumps(dates, indent=2), encoding="utf-8")
    if dates:
        latest_df = df[df["eom"] == dates[0]["date"]].sort_values("id")
        _write_json(pages_root / "latest.json", latest_df)

    stock_index = []
    for stock_id, stock_df in df.groupby("id", sort=True):
        stock_df = stock_df.sort_values("eom")
        first = stock_df.iloc[-1]
        payload = {
            "id": int(stock_id),
            "ticker": first.get("ticker", ""),
            "name": first.get("name", ""),
            "series": [
                [
                    row.eom,
                    None if pd.isna(row.y) else float(row.y),
                    None if pd.isna(row.v) else float(row.v),
                    None if pd.isna(row.e) else float(row.e),
                ]
                for row in stock_df.itertuples(index=False)
            ],
        }
        (stocks_dir / f"{stock_id}.json").write_text(
            json.dumps(payload, separators=(",", ":")),
            encoding="utf-8",
        )
        stock_index.append(
            {
                "id": int(stock_id),
                "ticker": first.get("ticker", ""),
                "name": first.get("name", ""),
                "observations": int(len(stock_df)),
            }
        )
    (pages_root / "stocks.json").write_text(
        json.dumps(stock_index, separators=(",", ":")),
        encoding="utf-8",
    )


def _write_json(path: Path, df: pd.DataFrame) -> None:
    records = df.where(pd.notnull(df), None).to_dict(orient="records")
    path.write_text(json.dumps(records, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main()
