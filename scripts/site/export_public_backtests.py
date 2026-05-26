from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.backtest import (
    aggregate_status_counts,
    aser_min_observations,
    aser_backtest_by_stock_r,
    cc_backtest_by_stock,
    overall_status,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export public backtesting JSON for GitHub Pages.")
    parser.add_argument("--predictions-root", default="docs/data/predictions")
    parser.add_argument("--output-root", default="docs/data/backtesting")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--skip-aser", action="store_true")
    parser.add_argument("--rscript-bin", default="Rscript")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions_root = Path(args.predictions_root)
    output_root = Path(args.output_root)
    stock_rows, series_df = load_public_prediction_series(predictions_root)

    cc = cc_backtest_by_stock(series_df, alpha=args.alpha)
    if args.skip_aser:
        aser = unavailable_aser(series_df)
    else:
        aser = aser_backtest_by_stock_r(
            series_df,
            alpha=args.alpha,
            version=2,
            rscript_bin=args.rscript_bin,
        )

    stock_payloads = build_stock_payloads(stock_rows, series_df, cc, aser)
    write_stock_payloads(output_root / "stocks", stock_payloads)
    write_json(output_root / "summary.json", build_summary(stock_payloads, series_df, args))
    write_json(output_root / "industry.json", build_industry_summary(stock_payloads))
    print(f"Exported backtests for {len(stock_payloads):,} stocks to {output_root}")


def load_public_prediction_series(predictions_root: Path) -> tuple[dict[int, dict], pd.DataFrame]:
    stocks = json.loads((predictions_root / "stocks.json").read_text(encoding="utf-8"))
    stock_rows = {int(row["id"]): row for row in stocks if has_display_ticker(row)}
    rows = []
    for stock_id in stock_rows:
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
                    "ticker": payload.get("ticker", ""),
                    "name": payload.get("name", ""),
                    "industry": payload.get("industry", ""),
                    "eom": item[0],
                    "target_month": next_month_end(item[0]),
                    "y": item[1],
                    "v": item[2],
                    "e": item[3],
                }
            )
    df = pd.DataFrame(rows)
    df["eom"] = pd.to_datetime(df["eom"], errors="coerce")
    df = df.dropna(subset=["id", "eom", "y", "v", "e"])
    return stock_rows, df


def build_stock_payloads(
    stock_rows: dict[int, dict],
    series_df: pd.DataFrame,
    cc: pd.DataFrame,
    aser: pd.DataFrame,
) -> list[dict]:
    cc_map = cc.set_index("id").to_dict(orient="index")
    aser_map = aser.set_index("id").to_dict(orient="index")
    latest = series_df.sort_values(["id", "eom"]).groupby("id", as_index=False).tail(1)
    latest_map = latest.set_index("id").to_dict(orient="index")

    payloads = []
    for stock_id, meta in stock_rows.items():
        cc_row = normalize_test_row(cc_map.get(stock_id, {}))
        aser_row = normalize_test_row(aser_map.get(stock_id, {}))
        latest_row = latest_map.get(stock_id, {})
        payloads.append(
            {
                "id": stock_id,
                "ticker": meta.get("ticker", ""),
                "name": meta.get("name", ""),
                "industry": meta.get("industry", ""),
                "latest_evaluated": format_date(latest_row.get("target_month")),
                "overall_status": overall_status(cc_row["status"], aser_row["status"]),
                "var_status": cc_row["status"],
                "es_status": aser_row["status"],
                "var_cc": cc_row,
                "es_aser": aser_row,
            }
        )
    return payloads


def normalize_test_row(row: dict) -> dict:
    return {
        "n_obs": clean_int(row.get("n_obs")),
        "statistic": clean_float(row.get("statistic")),
        "p_value": clean_float(row.get("p_value")),
        "status": row.get("status") or "unavailable",
        "hit_rate": clean_float(row.get("hit_rate")),
        "expected_hit_rate": clean_float(row.get("expected_hit_rate")),
    }


def build_summary(stock_payloads: list[dict], series_df: pd.DataFrame, args: argparse.Namespace) -> dict:
    latest_target = series_df["target_month"].max() if len(series_df) else None
    forecast_eom = series_df["eom"].max() if len(series_df) else None
    return {
        "latest_forecast": format_date(forecast_eom),
        "latest_evaluated": format_date(latest_target),
        "pending_window": "Updates when the next realized monthly return is added.",
        "stocks_evaluated": len(stock_payloads),
        "alpha": args.alpha,
        "aser_min_observations": aser_min_observations(args.alpha),
        "var_cc": aggregate_status_counts(stock_payloads, "var_status"),
        "es_aser": aggregate_status_counts(stock_payloads, "es_status"),
        "overall": aggregate_status_counts(stock_payloads, "overall_status"),
        "diagnostics": [
            "VaR: Christoffersen conditional coverage test",
            "ES: ASER test based on esback",
            f"ASER requires at least {aser_min_observations(args.alpha)} monthly observations under esback's default bandwidth.",
            "Status: Pass if p-value >= 5%; Fail if p-value < 5%",
        ],
        "reliability": [
            "Backtests use each stock's realized forecast history.",
            "New realized monthly returns are appended before rerunning tests.",
            "Live e-backtesting monitor is reserved for a later release.",
        ],
    }


def build_industry_summary(stock_payloads: list[dict]) -> list[dict]:
    rows = []
    for industry, group in pd.DataFrame(stock_payloads).groupby("industry", dropna=False):
        items = group.to_dict(orient="records")
        rows.append(
            {
                "industry": industry or "Unclassified",
                "stocks": len(items),
                "overall": aggregate_status_counts(items, "overall_status"),
                "var_cc": aggregate_status_counts(items, "var_status"),
                "es_aser": aggregate_status_counts(items, "es_status"),
            }
        )
    return sorted(rows, key=lambda row: row["stocks"], reverse=True)


def write_stock_payloads(stocks_dir: Path, stock_payloads: list[dict]) -> None:
    stocks_dir.mkdir(parents=True, exist_ok=True)
    for payload in stock_payloads:
        write_json(stocks_dir / f"{payload['id']}.json", payload)


def unavailable_aser(series_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    min_obs = aser_min_observations(0.05)
    for stock_id, group in series_df.groupby("id", sort=False):
        n_obs = len(group.dropna(subset=["y", "v", "e"]))
        rows.append(
            {
                "id": int(stock_id),
                "n_obs": int(n_obs),
                "statistic": np.nan,
                "p_value": np.nan,
                "status": "not_testable" if n_obs < min_obs else "unavailable",
            }
        )
    return pd.DataFrame(rows)


def has_display_ticker(row: dict) -> bool:
    ticker = str(row.get("ticker", "")).strip().upper()
    return bool(ticker) and ticker not in {"NA", "N/A"}


def next_month_end(date_value: str) -> str:
    return (pd.Timestamp(date_value) + pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d")


def format_date(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def clean_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def clean_int(value: object) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
