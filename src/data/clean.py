from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

from data.features import feature_names, label_name
from data.time_features import time_features


def process_feature(sub_df: pd.DataFrame, feature: str) -> pd.DataFrame:
    df_feature = sub_df.copy()
    df_feature[feature] = df_feature.groupby("eom")[feature].transform(
        lambda x: x.rank(method="min", pct=True) * 2 - 1
    )
    df_feature[feature] = df_feature.groupby("eom")[feature].transform(
        lambda x: x.fillna(x.median())
    )
    df_feature = df_feature.sort_values(["id", "eom"])
    df_feature[feature] = df_feature.groupby("id")[feature].ffill()
    return df_feature


def clean_country_data(
    raw_path: str | Path,
    output_path: str | Path,
    n_jobs: int = -1,
) -> Path:
    raw_path = Path(raw_path)
    output_path = Path(output_path)
    df = pd.read_csv(raw_path)
    df["eom"] = pd.to_datetime(df["eom"], format="%Y-%m-%d")
    base = df[["eom", "id", label_name]].copy()

    processed_features = Parallel(n_jobs=n_jobs)(
        delayed(process_feature)(df[["eom", "id", col]], col)
        for col in tqdm(feature_names, desc=f"Cleaning features from {raw_path.name}")
    )

    for col, proc_df in tqdm(list(zip(feature_names, processed_features)), desc="Merging features"):
        base = base.merge(proc_df, on=["eom", "id"], how="left")

    base.dropna(subset=[label_name], inplace=True)
    base[feature_names] = base[feature_names].fillna(0)
    base = base.set_index(["id", "eom"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(output_path)
    return output_path


def _filter_country_dates(df: pd.DataFrame, country: str, start: str | None, end: str | None) -> pd.DataFrame:
    out = df.copy()
    out["eom"] = pd.to_datetime(out["eom"])
    if start:
        out = out[out["eom"] >= pd.Timestamp(start)]
    elif country != "USA":
        out = out[out["eom"] >= pd.Timestamp("2013-02-01")]
    if end:
        out = out[out["eom"] <= pd.Timestamp(end)]
    elif country != "USA":
        out = out[out["eom"] <= pd.Timestamp("2024-12-31")]
    return out


def build_pointwise_samples(
    processed_path: str | Path,
    output_dir: str | Path,
    country: str,
    start: str | None = None,
    end: str | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = _filter_country_dates(pd.read_csv(processed_path), country, start, end)
    data["eom"] = pd.to_datetime(data["eom"])

    for row in tqdm(data.itertuples(index=False), total=len(data), desc=f"Pointwise samples {country}"):
        stock_id = int(getattr(row, "id"))
        eom = pd.Timestamp(getattr(row, "eom"))
        feature = np.array([getattr(row, col) for col in feature_names], dtype=np.float32)
        label = np.array([getattr(row, label_name)], dtype=np.float32)
        if np.isnan(feature).any() or np.isnan(label).any():
            continue
        np.save(
            output_dir / f"{stock_id}_{eom.strftime('%Y-%m-%d')}.npy",
            {
                "feature": feature,
                "label": label,
            },
        )
    return output_dir


def build_temporal_samples(
    processed_path: str | Path,
    output_dir: str | Path,
    country: str,
    sequence_length: int,
    start: str | None = None,
    end: str | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = _filter_country_dates(pd.read_csv(processed_path), country, start, end)

    for stock_id in tqdm(data["id"].unique(), desc=f"Temporal samples {country}"):
        stock_df = data[data["id"] == stock_id].copy().sort_values("eom")
        stock_df = stock_df.set_index("eom")
        full_dates = pd.date_range(stock_df.index.min(), stock_df.index.max(), freq="ME")
        stock_df = stock_df.reindex(full_dates)
        stock_df["id"] = stock_id
        stock_df = stock_df.reset_index().rename(columns={"index": "eom"})

        feature_array = stock_df[feature_names].values
        label_array = stock_df[label_name].values
        date_array = stock_df["eom"].values
        if len(feature_array) < sequence_length:
            continue

        for i in range(len(feature_array) - sequence_length + 1):
            x_enc = feature_array[i : i + sequence_length]
            y = label_array[i + sequence_length - 1]
            end_time = pd.Timestamp(date_array[i + sequence_length - 1])
            if np.isnan(x_enc).any() or np.isnan(y):
                continue

            x_dec = np.zeros_like(x_enc)
            x_dec[0] = x_enc[-1]
            enc_dates = pd.date_range(end=end_time, periods=sequence_length, freq="ME")
            dec_dates = pd.date_range(start=end_time + pd.offsets.MonthEnd(1), periods=2, freq="ME")
            sample = {
                "feature": x_enc.astype(np.float32),
                "label": np.array([y], dtype=np.float32),
                "x_dec": x_dec.astype(np.float32),
                "x_mark_enc": time_features(enc_dates, timeenc=0, freq="b").astype(np.float32),
                "x_mark_dec": time_features(dec_dates, timeenc=0, freq="b").astype(np.float32),
            }
            np.save(output_dir / f"{int(stock_id)}_{end_time.strftime('%Y-%m-%d')}.npy", sample)
    return output_dir


def build_cross_sectional_samples(
    processed_path: str | Path,
    temporal_dir: str | Path,
    output_dir: str | Path,
    country: str,
    start: str | None = None,
    end: str | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = _filter_country_dates(pd.read_csv(processed_path), country, start, end)
    stocks = df["id"].unique()
    dates = sorted(df["eom"].astype(str).unique())
    temporal_dir = Path(temporal_dir)

    for date in tqdm(dates, desc=f"Cross-sectional samples {country}"):
        features, labels, valid_ids = [], [], []
        for stock_id in stocks:
            sample_path = temporal_dir / f"{int(stock_id)}_{date}.npy"
            if not sample_path.exists():
                continue
            try:
                sample = np.load(sample_path, allow_pickle=True).item()
            except Exception:
                continue
            features.append(sample["feature"])
            labels.append(sample["label"])
            valid_ids.append(int(stock_id))
        if not valid_ids:
            continue
        np.save(
            output_dir / f"{date}.npy",
            {
                "feature": np.stack(features).astype(np.float32),
                "label": np.stack(labels).astype(np.float32),
                "stock_ids": np.array(valid_ids),
            },
        )
    return output_dir


def build_resga_samples(
    processed_path: str | Path,
    temporal_dir: str | Path,
    output_dir: str | Path,
    country: str,
    sequence_length: int,
    start: str | None = None,
    end: str | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = _filter_country_dates(pd.read_csv(processed_path), country, start, end)
    window_size = 5 * sequence_length
    df["eom"] = pd.to_datetime(df["eom"])
    df = df.sort_values(["eom", "id"])
    date_list = df["eom"].sort_values().unique()
    valid_dates = date_list[window_size - 1 :]
    stocks = df["id"].unique()
    temporal_dir = Path(temporal_dir)

    for date in tqdm(valid_dates, desc=f"Retrieval samples {country}"):
        valid_ids = [
            int(stock_id)
            for stock_id in stocks
            if (temporal_dir / f"{int(stock_id)}_{pd.Timestamp(date).strftime('%Y-%m-%d')}.npy").exists()
        ]
        if not valid_ids:
            continue
        idx = date_list.searchsorted(date)
        past_dates = date_list[idx - window_size + 1 : idx + 1]
        window_df = df[df["eom"].isin(past_dates) & df["id"].isin(valid_ids)]
        window_df = window_df[["id", "eom"] + feature_names]
        window_df = window_df.set_index(["id", "eom"]).unstack("eom")
        multi_col = pd.MultiIndex.from_product([feature_names, past_dates])
        window_df = window_df.reindex(columns=multi_col).reindex(index=valid_ids)

        values = window_df.values
        mask = np.isnan(values)
        medians = np.nanmedian(values, axis=0, keepdims=True)
        medians = np.where(np.isnan(medians), 0, medians)
        values[mask] = np.broadcast_to(medians, values.shape)[mask]

        n_stocks = len(window_df)
        n_features = len(feature_names)
        feature_tensor = torch.tensor(
            values.reshape(n_stocks, n_features, window_size).transpose(0, 2, 1),
            dtype=torch.float32,
        )
        target_df = df[(df["eom"] == date) & (df["id"].isin(valid_ids))]
        target_df = target_df.set_index("id").reindex(valid_ids).reset_index()
        sample = {
            "feature": feature_tensor,
            "label": torch.tensor(target_df[label_name].values, dtype=torch.float32),
            "stock_ids": torch.tensor(target_df["id"].values, dtype=torch.int),
        }
        torch.save(sample, output_dir / f"{pd.Timestamp(date).strftime('%Y%m%d')}.pt")
    return output_dir
