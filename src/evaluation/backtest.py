"""Backtesting utilities for public VaR/ES forecast diagnostics."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

ASER_BANDWIDTH_TAU = 0.05


def kupiec_lr_uc(var: np.ndarray, y: np.ndarray, alpha: float) -> float:
    """Kupiec unconditional coverage statistic for left-tail VaR hits."""
    hits = y <= var
    total = len(hits)
    if total == 0:
        return np.nan
    n_hits = int(hits.sum())
    p_hat = np.clip(n_hits / total, 1e-12, 1 - 1e-12)
    alpha = np.clip(alpha, 1e-12, 1 - 1e-12)
    ll_null = (total - n_hits) * np.log(1 - alpha) + n_hits * np.log(alpha)
    ll_alt = (total - n_hits) * np.log(1 - p_hat) + n_hits * np.log(p_hat)
    return float(-2 * (ll_null - ll_alt))


def christoffersen_cc(var: np.ndarray, y: np.ndarray, alpha: float) -> tuple[float, float]:
    """Christoffersen conditional coverage test for left-tail VaR hits."""
    hits = (y <= var).astype(float)
    total = len(hits)
    if total < 2:
        return np.nan, np.nan

    transitions = hits[1:] - hits[:-1]
    t01 = np.sum(transitions == 1.0)
    t10 = np.sum(transitions == -1.0)
    t00 = np.sum((transitions == 0.0) & (hits[:-1] == 0.0))
    t11 = np.sum((transitions == 0.0) & (hits[:-1] == 1.0))

    pi = np.clip((t01 + t11) / total, 1e-12, 1 - 1e-12)
    pi0 = np.clip(t01 / (t00 + t01), 1e-12, 1 - 1e-12) if (t00 + t01) else 1e-12
    pi1 = np.clip(t11 / (t10 + t11), 1e-12, 1 - 1e-12) if (t10 + t11) else 1e-12

    ll_null = (t00 + t10) * np.log(1 - pi) + (t01 + t11) * np.log(pi)
    ll_alt = t00 * np.log(1 - pi0) + t01 * np.log(pi0) + t10 * np.log(1 - pi1) + t11 * np.log(pi1)
    lr_ind = -2 * (ll_null - ll_alt)
    lr_cc = float(lr_ind + kupiec_lr_uc(var, y, alpha))
    return lr_cc, float(1 - chi2.cdf(lr_cc, df=2))


def cc_backtest_by_stock(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Run VaR CC test by stock id."""
    rows = []
    for stock_id, group in df.groupby("id", sort=False):
        clean = _clean_group(group, ["y", "v"])
        if len(clean) < 2:
            rows.append(_empty_result(stock_id, len(clean), "not_testable"))
            continue
        stat, p_value = christoffersen_cc(clean["v"].to_numpy(), clean["y"].to_numpy(), alpha)
        hit_rate = float((clean["y"] <= clean["v"]).mean())
        rows.append(
            {
                "id": int(stock_id),
                "n_obs": int(len(clean)),
                "statistic": stat,
                "p_value": p_value,
                "status": status_from_pvalue(p_value),
                "hit_rate": hit_rate,
                "expected_hit_rate": alpha,
            }
        )
    return pd.DataFrame(rows)


def aser_backtest_by_stock_r(
    df: pd.DataFrame,
    alpha: float = 0.05,
    version: int = 2,
    rscript_bin: str = "Rscript",
) -> pd.DataFrame:
    """Run ASER/ESR tests using R esback::esr_backtest in a single R process."""
    required = ["id", "eom", "y", "v", "e"]
    clean = df[required].replace([np.inf, -np.inf], np.nan).dropna().copy()
    clean = clean.sort_values(["id", "eom"])
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        input_path = tmpdir_path / "input.csv"
        output_path = tmpdir_path / "output.csv"
        script_path = tmpdir_path / "aser_backtest.R"
        clean.to_csv(input_path, index=False)
        script_path.write_text(_aser_r_script(), encoding="utf-8")
        cmd = [
            rscript_bin,
            str(script_path),
            str(input_path),
            str(output_path),
            str(alpha),
            str(version),
            str(aser_min_observations(alpha)),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            return _aser_unavailable(df, str(exc))
        if not output_path.exists():
            return _aser_unavailable(df, "R output was not created.")
        return pd.read_csv(output_path)


def status_from_pvalue(p_value: float | None) -> str:
    if p_value is None or not np.isfinite(p_value):
        return "unavailable"
    return "pass" if p_value >= 0.05 else "fail"


def aggregate_status_counts(rows: list[dict], key: str) -> dict[str, int]:
    counts = {"pass": 0, "fail": 0, "not_testable": 0, "unavailable": 0}
    for row in rows:
        status = row.get(key, "unavailable")
        counts[status] = counts.get(status, 0) + 1
    return counts


def overall_status(*statuses: str) -> str:
    rank = {"fail": 0, "pass": 1, "not_testable": 2, "unavailable": 3}
    valid = [status for status in statuses if status in rank]
    if not valid:
        return "unavailable"
    return min(valid, key=lambda status: rank[status])


def _clean_group(group: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return group.sort_values("eom").replace([np.inf, -np.inf], np.nan).dropna(subset=columns)


def _empty_result(stock_id: int, n_obs: int, status: str) -> dict:
    return {
        "id": int(stock_id),
        "n_obs": int(n_obs),
        "statistic": np.nan,
        "p_value": np.nan,
        "status": status,
    }


def aser_min_observations(alpha: float) -> int:
    """Minimum n needed by esreg's default Hall-Sheather sparsity bandwidth."""
    constant = (
        norm_ppf(1 - ASER_BANDWIDTH_TAU / 2) ** (2 / 3)
        * ((1.5 * norm_pdf(norm_ppf(alpha)) ** 2) / (2 * norm_ppf(alpha) ** 2 + 1)) ** (1 / 3)
    )
    return int(np.floor((constant / alpha) ** 3) + 1)


def norm_ppf(value: float) -> float:
    return float(norm.ppf(value))


def norm_pdf(value: float) -> float:
    return float(norm.pdf(value))


def _aser_unavailable(df: pd.DataFrame, reason: str) -> pd.DataFrame:
    rows = []
    for stock_id, group in df.groupby("id", sort=False):
        row = _empty_result(stock_id, len(group.dropna(subset=["y", "v", "e"])), "unavailable")
        row["reason"] = reason
        rows.append(row)
    return pd.DataFrame(rows)


def _aser_r_script() -> str:
    return r"""
suppressPackageStartupMessages(library(esback))
args <- commandArgs(trailingOnly = TRUE)
input_path <- args[[1]]
output_path <- args[[2]]
alpha <- as.numeric(args[[3]])
version <- as.integer(args[[4]])
min_obs <- as.integer(args[[5]])
df <- read.csv(input_path, stringsAsFactors = FALSE)
ids <- unique(df$id)
out <- vector("list", length(ids))
for (i in seq_along(ids)) {
  sid <- ids[[i]]
  s <- df[df$id == sid, ]
  s <- s[order(s$eom), ]
  s <- s[is.finite(s$y) & is.finite(s$v) & is.finite(s$e), ]
  if (nrow(s) < min_obs) {
    out[[i]] <- data.frame(id=sid, n_obs=nrow(s), statistic=NA_real_, p_value=NA_real_, status="not_testable")
    next
  }
  result <- tryCatch(
    esr_backtest(r=s$y, q=s$v, e=s$e, alpha=alpha, version=version, B=0),
    error=function(e) NULL
  )
  if (is.null(result)) {
    out[[i]] <- data.frame(id=sid, n_obs=nrow(s), statistic=NA_real_, p_value=NA_real_, status="unavailable")
    next
  }
  pval <- as.numeric(result$pvalue_twosided_asymptotic)
  stat <- if ("statistic" %in% names(result)) as.numeric(result$statistic) else NA_real_
  status <- if (is.na(pval)) "unavailable" else if (pval >= 0.05) "pass" else "fail"
  out[[i]] <- data.frame(id=sid, n_obs=nrow(s), statistic=stat, p_value=pval, status=status)
}
write.csv(do.call(rbind, out), output_path, row.names=FALSE)
"""


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False), encoding="utf-8")
