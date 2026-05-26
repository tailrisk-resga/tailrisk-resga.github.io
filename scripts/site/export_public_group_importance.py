from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.data.groups import GROUPS
except Exception:  # pragma: no cover - keeps this script runnable beside old research code.
    GROUPS = {}


CAP_GROUPS = ["ALL", "mega", "large", "small", "micro", "nano"]
REQUIRED_COLUMNS = {"id", "eom", "mask", "y", "v", "e"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate ReSGA feature-group ablation outputs for the public dashboard."
    )
    parser.add_argument(
        "--input-root",
        default="outputs/source_results/USA",
        help="Root containing <model>/seed_<seed>/<hyperparameter>/<window>/group_importance.csv.",
    )
    parser.add_argument("--model", default="Retrieval")
    parser.add_argument("--hyperparameter", default="1_512_10")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument(
        "--size-group-path",
        default="data/metadata/USA_size_grp.csv",
        help="CSV with id, eom, and size_grp columns.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/public_site_data/USA/group_importance/1_512_10",
        help="Directory where archive CSV/JSON files will be written.",
    )
    parser.add_argument(
        "--github-pages-root",
        default=None,
        help="Optional docs/data/group_importance directory to update with compact website JSON.",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--normalize",
        choices=["positive", "relative_positive", "softmax", "power", "log"],
        default="positive",
        help="Importance normalization method. The paper figures used positive by default.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=10, help="Number of top groups saved in summary fields.")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow id/eom/mask rows with fewer than all requested seeds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    aggregated = aggregate_group_predictions(
        input_root=Path(args.input_root),
        model=args.model,
        hyperparameter=args.hyperparameter,
        seeds=args.seeds,
        size_group_path=Path(args.size_group_path),
        alpha=args.alpha,
        allow_incomplete=args.allow_incomplete,
    )
    overall = group_importance_overall_loss(
        aggregated,
        normalize=args.normalize,
        temperature=args.temperature,
        gamma=args.gamma,
    )
    monthly = group_importance_by_month(
        aggregated,
        normalize=args.normalize,
        temperature=args.temperature,
        gamma=args.gamma,
    )
    metadata = build_metadata(args, aggregated, overall, monthly)

    write_archive(output_root, aggregated, overall, monthly, metadata)
    if args.github_pages_root:
        write_pages_data(Path(args.github_pages_root), overall, monthly, metadata)

    print(f"Aggregated {len(aggregated):,} group-importance rows.")
    print(f"Wrote archive data to {output_root}")
    if args.github_pages_root:
        print(f"Wrote website data to {args.github_pages_root}")


def aggregate_group_predictions(
    input_root: Path,
    model: str,
    hyperparameter: str,
    seeds: list[int],
    size_group_path: Path,
    alpha: float,
    allow_incomplete: bool = False,
) -> pd.DataFrame:
    frames = []
    expected_windows: set[str] | None = None

    for seed in seeds:
        seed_dir = input_root / model / f"seed_{seed}" / hyperparameter
        paths = sorted(seed_dir.glob("*/group_importance.csv"))
        if not paths:
            raise FileNotFoundError(f"No group_importance.csv files found under {seed_dir}")

        windows = {path.parent.name for path in paths}
        if expected_windows is None:
            expected_windows = windows
        elif windows != expected_windows:
            missing = sorted(expected_windows - windows)
            extra = sorted(windows - expected_windows)
            raise ValueError(f"seed_{seed} windows mismatch; missing={missing}, extra={extra}")

        for path in paths:
            df = pd.read_csv(path)
            missing = sorted(REQUIRED_COLUMNS - set(df.columns))
            if missing:
                raise ValueError(f"{path} is missing required columns: {missing}")
            df = df[list(REQUIRED_COLUMNS)].copy()
            df["seed"] = seed
            df["window"] = path.parent.name
            frames.append(df)

    all_rows = pd.concat(frames, ignore_index=True)
    all_rows["eom"] = pd.to_datetime(all_rows["eom"], errors="coerce").dt.strftime("%Y-%m-%d")
    all_rows = all_rows.dropna(subset=["id", "eom", "mask", "y", "v", "e"])

    grouped = (
        all_rows.groupby(["id", "eom", "mask"], as_index=False)
        .agg(
            y=("y", "mean"),
            v=("v", "mean"),
            e=("e", "mean"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(["eom", "id", "mask"])
    )

    seed_count = len(seeds)
    incomplete = grouped.loc[grouped["n_seeds"] != seed_count]
    if not incomplete.empty and not allow_incomplete:
        raise ValueError(
            f"{len(incomplete):,} id/eom/mask rows do not have all {seed_count} seeds. "
            "Rerun with --allow-incomplete only if this is intentional."
        )

    size_groups = pd.read_csv(size_group_path)
    missing_size_cols = sorted({"id", "eom", "size_grp"} - set(size_groups.columns))
    if missing_size_cols:
        raise ValueError(f"{size_group_path} is missing required columns: {missing_size_cols}")
    size_groups = size_groups[["id", "eom", "size_grp"]].copy()
    size_groups["eom"] = pd.to_datetime(size_groups["eom"], errors="coerce").dt.strftime("%Y-%m-%d")

    merged = grouped.merge(size_groups, on=["id", "eom"], how="inner")
    if merged.empty:
        raise ValueError("No rows remain after merging with size groups. Check id/eom alignment.")

    merged["loss"] = fz0_loss_values(merged["y"], merged["v"], merged["e"], alpha=alpha)
    merged["target_month"] = (
        pd.to_datetime(merged["eom"], errors="coerce") + pd.offsets.MonthEnd(1)
    ).dt.strftime("%Y-%m-%d")
    return merged.sort_values(["eom", "id", "mask"]).reset_index(drop=True)


def group_importance_overall_loss(
    df: pd.DataFrame,
    normalize: str,
    temperature: float = 1.0,
    gamma: float = 0.5,
) -> pd.DataFrame:
    rows = []
    for cap_group in CAP_GROUPS:
        group_df = df if cap_group == "ALL" else df.loc[df["size_grp"] == cap_group]
        if group_df.empty:
            continue

        loss_df = group_df.groupby("mask", as_index=False)["loss"].mean()
        baseline = baseline_loss(loss_df, cap_group)
        loss_df["baseline_loss"] = baseline
        loss_df["delta_abs"] = loss_df["loss"] - baseline
        loss_df["delta_rel"] = loss_df["delta_abs"] / abs(baseline) if baseline else np.nan

        non_baseline = loss_df["mask"] != "ALL"
        loss_df["importance"] = 0.0
        loss_df.loc[non_baseline, "importance"] = normalize_importance(
            loss_df.loc[non_baseline, "delta_abs"].to_numpy(),
            loss_df.loc[non_baseline, "delta_rel"].to_numpy(),
            mode=normalize,
            temperature=temperature,
            gamma=gamma,
        )
        loss_df["cap_group"] = cap_group
        loss_df["rank"] = rank_importance(loss_df)
        rows.append(loss_df)

    return reorder_columns(pd.concat(rows, ignore_index=True), include_month=False)


def group_importance_by_month(
    df: pd.DataFrame,
    normalize: str,
    temperature: float = 1.0,
    gamma: float = 0.5,
) -> pd.DataFrame:
    rows = []
    month_df = df.copy()
    month_df["eom"] = pd.to_datetime(month_df["eom"], errors="coerce")
    month_df = month_df.dropna(subset=["eom"])

    for eom, one_month in month_df.groupby("eom", sort=True):
        loss_df = one_month.groupby("mask", as_index=False)["loss"].mean()
        baseline = baseline_loss(loss_df, str(eom.date()))
        loss_df["baseline_loss"] = baseline
        loss_df["delta_abs"] = loss_df["loss"] - baseline
        loss_df["delta_rel"] = loss_df["delta_abs"] / abs(baseline) if baseline else np.nan

        non_baseline = loss_df["mask"] != "ALL"
        loss_df["importance"] = 0.0
        loss_df.loc[non_baseline, "importance"] = normalize_importance(
            loss_df.loc[non_baseline, "delta_abs"].to_numpy(),
            loss_df.loc[non_baseline, "delta_rel"].to_numpy(),
            mode=normalize,
            temperature=temperature,
            gamma=gamma,
        )
        total = loss_df.loc[non_baseline, "importance"].sum()
        if total > 0:
            loss_df.loc[non_baseline, "importance"] /= total
        loss_df["eom"] = eom.strftime("%Y-%m-%d")
        loss_df["target_month"] = (eom + pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d")
        loss_df["rank"] = rank_importance(loss_df)
        rows.append(loss_df)

    return reorder_columns(pd.concat(rows, ignore_index=True), include_month=True)


def fz0_loss_values(y: pd.Series, v: pd.Series, e: pd.Series, alpha: float) -> np.ndarray:
    y_arr = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    v_arr = pd.to_numeric(v, errors="coerce").to_numpy(dtype=float)
    e_arr = pd.to_numeric(e, errors="coerce").to_numpy(dtype=float)

    invalid = (~np.isfinite(e_arr)) | (e_arr >= 0)
    safe_e = e_arr.copy()
    safe_e[invalid] = np.nan
    indicator = (y_arr <= v_arr).astype(float)
    loss = -(1.0 / (alpha * safe_e)) * indicator * (v_arr - y_arr) + (v_arr / safe_e) + np.log(-100.0 * safe_e) - 1.0
    return loss


def baseline_loss(loss_df: pd.DataFrame, label: str) -> float:
    baseline = loss_df.loc[loss_df["mask"] == "ALL", "loss"]
    if baseline.empty:
        raise ValueError(f"Missing baseline mask='ALL' for {label}")
    return float(baseline.iloc[0])


def normalize_importance(
    deltas_abs: np.ndarray,
    deltas_rel: np.ndarray,
    mode: str,
    temperature: float = 1.0,
    gamma: float = 0.5,
) -> np.ndarray:
    if mode == "positive":
        values = np.clip(deltas_abs, 0, None)
        total = values.sum()
        return values / total if total > 0 else np.zeros_like(values)

    if mode == "relative_positive":
        values = np.clip(deltas_rel, 0, None)
        total = values.sum()
        return values / total if total > 0 else np.zeros_like(values)

    if mode == "softmax":
        values = deltas_abs / max(float(temperature), 1e-12)
        values = values - np.nanmax(values)
        exp_values = np.exp(values)
        total = exp_values.sum()
        return exp_values / total if total > 0 else np.zeros_like(values)

    if mode == "power":
        values = np.clip(deltas_abs, 0, None) ** gamma
        total = values.sum()
        return values / total if total > 0 else np.zeros_like(values)

    if mode == "log":
        values = np.log1p(np.clip(deltas_abs, 0, None))
        total = values.sum()
        return values / total if total > 0 else np.zeros_like(values)

    raise ValueError(f"Unknown normalization mode: {mode}")


def rank_importance(df: pd.DataFrame) -> pd.Series:
    ranks = pd.Series(np.nan, index=df.index)
    mask = df["mask"] != "ALL"
    ranks.loc[mask] = df.loc[mask, "importance"].rank(method="first", ascending=False)
    return ranks


def reorder_columns(df: pd.DataFrame, include_month: bool) -> pd.DataFrame:
    numeric_cols = ["loss", "baseline_loss", "delta_abs", "delta_rel", "importance", "rank"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if include_month:
        columns = ["eom", "target_month", "mask", *numeric_cols]
    else:
        columns = ["cap_group", "mask", *numeric_cols]
    return df[columns].sort_values(columns[:1] + ["rank", "mask"]).reset_index(drop=True)


def build_metadata(
    args: argparse.Namespace,
    aggregated: pd.DataFrame,
    overall: pd.DataFrame,
    monthly: pd.DataFrame,
) -> dict:
    latest_eom = aggregated["eom"].max()
    latest_target = aggregated.loc[aggregated["eom"] == latest_eom, "target_month"].max()
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "hyperparameter": args.hyperparameter,
        "seeds": args.seeds,
        "alpha": args.alpha,
        "normalize": args.normalize,
        "latest_eom": latest_eom,
        "latest_target_month": latest_target,
        "observations": {
            "rows": int(len(aggregated)),
            "stocks": int(aggregated["id"].nunique()),
            "months": int(aggregated["eom"].nunique()),
            "groups": int((aggregated["mask"] != "ALL").sum() / max(aggregated[["id", "eom"]].drop_duplicates().shape[0], 1)),
        },
        "groups": group_metadata(overall),
        "summary": {
            "overall_top": top_records(overall.loc[overall["cap_group"] == "ALL"], args.top_k),
            "by_size_top": {
                cap: top_records(overall.loc[overall["cap_group"] == cap], args.top_k)
                for cap in CAP_GROUPS
                if cap in set(overall["cap_group"])
            },
            "latest_month_top": top_records(monthly.loc[monthly["eom"] == monthly["eom"].max()], args.top_k),
        },
    }


def group_metadata(overall: pd.DataFrame) -> list[dict]:
    masks = sorted(mask for mask in overall["mask"].unique() if mask != "ALL")
    return [
        {
            "mask": mask,
            "display_name": display_group_name(mask),
            "feature_count": int(len(GROUPS.get(mask, []))),
        }
        for mask in masks
    ]


def top_records(df: pd.DataFrame, top_k: int) -> list[dict]:
    filtered = df.loc[df["mask"] != "ALL"].sort_values(["importance", "mask"], ascending=[False, True]).head(top_k)
    records = []
    for row in filtered.itertuples(index=False):
        records.append(
            {
                "mask": row.mask,
                "display_name": display_group_name(row.mask),
                "importance": clean_float(row.importance),
                "delta_abs": clean_float(row.delta_abs),
                "rank": clean_float(row.rank),
            }
        )
    return records


def write_archive(
    output_root: Path,
    aggregated: pd.DataFrame,
    overall: pd.DataFrame,
    monthly: pd.DataFrame,
    metadata: dict,
) -> None:
    aggregated.to_csv(output_root / "aggregated_group_predictions.csv", index=False)
    overall.to_csv(output_root / "overall_importance.csv", index=False)
    monthly.to_csv(output_root / "monthly_importance.csv", index=False)
    write_json(output_root / "metadata.json", metadata, indent=2)


def write_pages_data(pages_root: Path, overall: pd.DataFrame, monthly: pd.DataFrame, metadata: dict) -> None:
    pages_root.mkdir(parents=True, exist_ok=True)
    overall_records = records_for_json(overall)
    monthly_records = records_for_json(monthly)
    write_json(pages_root / "overall.json", overall_records)
    write_json(pages_root / "monthly.json", monthly_records)
    write_json(
        pages_root / "latest.json",
        {
            **metadata,
            "overall": overall_records,
            "monthly": monthly_records,
        },
    )


def records_for_json(df: pd.DataFrame) -> list[dict]:
    records = df.where(pd.notnull(df), None).to_dict(orient="records")
    for row in records:
        if "rank" in row and row["rank"] is not None:
            row["rank"] = int(row["rank"])
        if "mask" in row:
            row["display_name"] = display_group_name(row["mask"])
            row["feature_count"] = int(len(GROUPS.get(row["mask"], [])))
    return records


def write_json(path: Path, payload: object, indent: int | None = None) -> None:
    path.write_text(json.dumps(clean_for_json(payload), ensure_ascii=False, indent=indent, separators=None if indent else (",", ":")), encoding="utf-8")


def clean_for_json(value: object) -> object:
    if isinstance(value, dict):
        return {key: clean_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [clean_for_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return clean_float(value)
    if isinstance(value, float):
        return clean_float(value)
    return value


def clean_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def display_group_name(mask: str) -> str:
    names = {
        "ShortTermReversal": "Short-Term Reversal",
        "LowRisk": "Low Risk",
        "DebtIssuance": "Debt Issuance",
        "LowLeverage": "Low Leverage",
        "ProfitGrowth": "Profit Growth",
    }
    if mask in names:
        return names[mask]
    return re.sub(r"(?<!^)(?=[A-Z])", " ", mask).strip()


if __name__ == "__main__":
    main()
