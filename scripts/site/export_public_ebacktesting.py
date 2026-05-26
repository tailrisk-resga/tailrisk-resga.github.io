from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.e_backtest import aggregate_alert_counts, ebacktest_one_stock, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export public e-backtesting JSON for GitHub Pages.")
    parser.add_argument("--predictions-root", default="docs/data/predictions")
    parser.add_argument("--output-root", default="docs/data/e_backtesting")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--grid-size", type=int, default=201)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions_root = Path(args.predictions_root)
    output_root = Path(args.output_root)

    stock_rows, series_df = load_public_prediction_series(predictions_root)
    stock_payloads = build_stock_payloads(stock_rows, series_df, args)
    write_stock_payloads(output_root / "stocks", stock_payloads)
    write_json(output_root / "summary.json", build_summary(stock_payloads, series_df, args))
    print(f"Exported e-backtesting monitor for {len(stock_payloads):,} stocks to {output_root}")


def load_public_prediction_series(predictions_root: Path) -> tuple[dict[int, dict], pd.DataFrame]:
    stocks = json.loads((predictions_root / "stocks.json").read_text(encoding="utf-8"))
    stock_rows = {int(row["id"]): row for row in stocks if has_display_ticker(row)}
    rows = []

    for stock_id, meta in stock_rows.items():
        path = predictions_root / "stocks" / f"{stock_id}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not has_display_ticker(payload):
            continue
        for item in payload.get("series", []):
            if len(item) < 4:
                continue
            rows.append(
                {
                    "id": stock_id,
                    "ticker": meta.get("ticker", ""),
                    "name": meta.get("name", ""),
                    "industry": meta.get("industry", ""),
                    "eom": item[0],
                    "target_month": next_month_end(item[0]),
                    "y": item[1],
                    "v": item[2],
                    "e": item[3],
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return stock_rows, df
    df["eom"] = pd.to_datetime(df["eom"], errors="coerce")
    df["target_month"] = pd.to_datetime(df["target_month"], errors="coerce")
    df = df.dropna(subset=["id", "eom", "target_month", "y", "v", "e"])
    return stock_rows, df


def build_stock_payloads(stock_rows: dict[int, dict], series_df: pd.DataFrame, args: argparse.Namespace) -> list[dict]:
    payloads = []
    for stock_id, meta in stock_rows.items():
        stock_df = series_df.loc[series_df["id"] == stock_id]
        monitor = ebacktest_one_stock(
            stock_df,
            alpha=args.alpha,
            gamma=args.gamma,
            grid_size=args.grid_size,
        )
        payloads.append(
            {
                "id": stock_id,
                "ticker": meta.get("ticker", ""),
                "name": meta.get("name", ""),
                "industry": meta.get("industry", ""),
                "alpha": args.alpha,
                "method": "GREE",
                "gamma": args.gamma,
                "history": "expanding",
                **monitor,
            }
        )
    return payloads


def build_summary(stock_payloads: list[dict], series_df: pd.DataFrame, args: argparse.Namespace) -> dict:
    counts = aggregate_alert_counts(stock_payloads)
    evaluated = counts["green"] + counts["yellow"] + counts["red"]
    red_share = counts["red"] / evaluated if evaluated else 0.0
    yellow_red_share = (counts["yellow"] + counts["red"]) / evaluated if evaluated else 0.0
    latest_target = series_df["target_month"].max() if len(series_df) else None

    return {
        "latest_evaluated": format_date(latest_target),
        "stocks_evaluated": len(stock_payloads),
        "alpha": args.alpha,
        "method": "GREE",
        "gamma": args.gamma,
        "history": "expanding",
        "thresholds": {
            "yellow": 2,
            "red": 20,
        },
        "alert_counts": counts,
        "red_share": red_share,
        "yellow_or_red_share": yellow_red_share,
        "description": "Sequential ES e-backtesting monitor updated with each newly realized monthly return.",
    }


def write_stock_payloads(stocks_dir: Path, stock_payloads: list[dict]) -> None:
    stocks_dir.mkdir(parents=True, exist_ok=True)
    for payload in stock_payloads:
        write_json(stocks_dir / f"{payload['id']}.json", payload)


def has_display_ticker(row: dict) -> bool:
    ticker = str(row.get("ticker", "")).strip().upper()
    return bool(ticker) and ticker not in {"NA", "N/A"}


def next_month_end(date_value: str) -> str:
    return (pd.Timestamp(date_value) + pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d")


def format_date(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


if __name__ == "__main__":
    main()
