"""E-backtesting utilities for live ES monitoring."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def es_e_statistic_from_returns(
    y: np.ndarray,
    v: np.ndarray,
    e: np.ndarray,
    alpha: float = 0.05,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute the ES e-statistic from return-convention y, VaR, and ES."""
    y = np.asarray(y, dtype=float).reshape(-1)
    v = np.asarray(v, dtype=float).reshape(-1)
    e = np.asarray(e, dtype=float).reshape(-1)
    if not (len(y) == len(v) == len(e)):
        raise ValueError("y, v, and e must have the same length.")

    loss = -y
    var_loss = -v
    es_loss = -e
    spread = es_loss - var_loss
    numerator = np.maximum(loss - var_loss, 0.0)
    denominator = float(alpha) * spread
    out = np.full(len(y), np.nan, dtype=float)
    valid = (
        np.isfinite(loss)
        & np.isfinite(var_loss)
        & np.isfinite(es_loss)
        & np.isfinite(denominator)
        & (denominator > eps)
    )
    out[valid] = numerator[valid] / denominator[valid]
    return out


def choose_lambda_gree(x_history: np.ndarray, gamma: float = 0.5, grid_size: int = 201) -> float:
    """Choose lambda by maximizing empirical log growth over past e-statistics."""
    x_history = np.asarray(x_history, dtype=float)
    x_history = x_history[np.isfinite(x_history)]
    if not len(x_history):
        return 0.0
    grid = np.linspace(0.0, float(gamma), int(grid_size))
    values = 1.0 - grid[:, None] + grid[:, None] * x_history[None, :]
    valid = np.all(values > 0, axis=1)
    scores = np.full(len(grid), -np.inf, dtype=float)
    scores[valid] = np.mean(np.log(values[valid]), axis=1)
    if not np.any(np.isfinite(scores)):
        return 0.0
    return float(grid[int(np.nanargmax(scores))])


def ebacktest_one_stock(
    stock_df: pd.DataFrame,
    alpha: float = 0.05,
    gamma: float = 0.5,
    grid_size: int = 201,
) -> dict:
    """Run expanding-history GREE e-backtesting for one stock."""
    clean = (
        stock_df.sort_values("eom")
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["eom", "y", "v", "e"])
        .copy()
    )
    if clean.empty:
        return empty_monitor_result()

    x_values = es_e_statistic_from_returns(
        clean["y"].to_numpy(),
        clean["v"].to_numpy(),
        clean["e"].to_numpy(),
        alpha=alpha,
    )
    valid = np.isfinite(x_values)
    clean = clean.loc[valid].reset_index(drop=True)
    x_values = x_values[valid]
    if clean.empty:
        return empty_monitor_result()

    lambdas = np.zeros(len(x_values), dtype=float)
    process = np.ones(len(x_values), dtype=float)
    for idx, x_value in enumerate(x_values):
        lam = choose_lambda_gree(x_values[:idx], gamma=gamma, grid_size=grid_size)
        lambdas[idx] = lam
        multiplier = 1.0 - lam + lam * x_value
        process[idx] = multiplier if idx == 0 else process[idx - 1] * multiplier

    max_m = float(np.max(process))
    final_m = float(process[-1])
    return {
        "n_obs": int(len(x_values)),
        "latest_evaluated": format_date(clean["target_month"].iloc[-1] if "target_month" in clean else clean["eom"].iloc[-1]),
        "final_m": final_m,
        "max_m": max_m,
        "alert": alert_level(max_m),
        "hit_2": first_hit(process, 2.0),
        "hit_5": first_hit(process, 5.0),
        "hit_10": first_hit(process, 10.0),
        "hit_20": first_hit(process, 20.0),
        "path": [
            [
                format_date(clean["target_month"].iloc[idx] if "target_month" in clean else clean["eom"].iloc[idx]),
                clean_float(x_values[idx]),
                clean_float(lambdas[idx]),
                clean_float(process[idx]),
            ]
            for idx in range(len(x_values))
        ],
    }


def empty_monitor_result() -> dict:
    return {
        "n_obs": 0,
        "latest_evaluated": "",
        "final_m": None,
        "max_m": None,
        "alert": "unavailable",
        "hit_2": None,
        "hit_5": None,
        "hit_10": None,
        "hit_20": None,
        "path": [],
    }


def alert_level(max_m: float | None) -> str:
    if max_m is None or not np.isfinite(max_m):
        return "unavailable"
    if max_m >= 20:
        return "red"
    if max_m >= 2:
        return "yellow"
    return "green"


def first_hit(values: np.ndarray, threshold: float) -> int | None:
    idx = np.where(values >= threshold)[0]
    return int(idx[0] + 1) if len(idx) else None


def aggregate_alert_counts(rows: list[dict]) -> dict[str, int]:
    counts = {"green": 0, "yellow": 0, "red": 0, "unavailable": 0}
    for row in rows:
        status = row.get("alert", "unavailable")
        counts[status] = counts.get(status, 0) + 1
    return counts


def clean_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def format_date(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False), encoding="utf-8")
