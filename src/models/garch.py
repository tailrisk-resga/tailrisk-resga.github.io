"""GARCH baseline."""

import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm
from tqdm import tqdm

from models.baseline_utils import TEST_END, TEST_YEARS, _prepare_training_series, sample_VE

# =========================
# Matlab-aligned GARCH-FZ (garch_FZ_LL.m) core
# =========================
def _transform_theta_garch_fz(theta_raw: np.ndarray):
    """
    Matlab garch_FZ_LL:
      beta = normcdf(theta(1))
      gamma = exp(theta(2))
      b = -exp(theta(3))
      c = normcdf(theta(4))
      a = c*b
    """
    theta_raw = np.asarray(theta_raw, dtype=float)
    beta  = norm.cdf(theta_raw[0])   # (0,1)
    gamma = np.exp(theta_raw[1])     # >0
    b     = -np.exp(theta_raw[2])    # <0
    c     = norm.cdf(theta_raw[3])   # (0,1)
    a     = c * b                    # ensures ES < VaR < 0 (since b<0, c in (0,1))
    return beta, gamma, a, b, c


def garch_FZ_LL_objective(theta_raw: np.ndarray,
                          data: np.ndarray,
                          alpha: float,
                          tau: float = -1.0,
                          omega: float = 1.0,
                          h0: float | None = None) -> float:
    """
    Matlab-aligned objective: average hom-deg-0 FZ loss + sanity penalties.

    Model:
      h_t = omega + beta*h_{t-1} + gamma*y_{t-1}^2
      VaR_t = a*sqrt(h_t), ES_t = b*sqrt(h_t)
      hit_t = I(y_t < VaR_t) (or smoothed via tau)

    Loss at time t:
      L_t = -(1/alpha)/ES_t * hit_t * (VaR_t - y_t) - (1/ES_t)*(ES_t - VaR_t) + log(-ES_t)

    Penalties (same spirit as Matlab):
      - ES < VaR and both < 0, and ES not too extreme vs realized minima
      - must see some hits
      - finite/real outputs only
    """
    data = np.asarray(data, dtype=float)
    T = data.size
    if T < 2 or np.any(~np.isfinite(data)):
        return 1e7

    beta, gamma, a, b, c = _transform_theta_garch_fz(theta_raw)

    if h0 is None:
        h0 = float(np.var(data))
    if (not np.isfinite(h0)) or h0 <= 0:
        h0 = 1.0

    # recurse variance and compute VE + loss
    hhat = np.empty(T, dtype=float)
    VEhat = np.empty((T, 2), dtype=float)
    loss = np.empty(T, dtype=float)

    hhat[0] = h0
    VEhat[0, :] = np.array([a, b]) * np.sqrt(hhat[0])

    # hit at t=0
    if tau == -1 or tau is None:
        hitS = 1.0 if (data[0] < VEhat[0, 0]) else 0.0
    else:
        # 1/(1+exp(tau*(y - VaR))) = expit(-tau*(y - VaR))
        hitS = expit(-tau * (data[0] - VEhat[0, 0]))

    ES0, VaR0 = VEhat[0, 1], VEhat[0, 0]
    if (not np.isfinite(ES0)) or ES0 >= 0:
        return 1e7
    loss[0] = -(1.0/alpha)/ES0 * hitS * (VaR0 - data[0]) - (1.0/ES0) * (ES0 - VaR0) + np.log(-ES0)

    for tt in range(1, T):
        hitSL = hitS
        hhat[tt] = omega + beta * hhat[tt-1] + gamma * (data[tt-1] ** 2)
        if (not np.isfinite(hhat[tt])) or hhat[tt] <= 0:
            return 1e7

        VEhat[tt, :] = np.array([a, b]) * np.sqrt(hhat[tt])

        if tau == -1 or tau is None:
            hitS = 1.0 if (data[tt] < VEhat[tt, 0]) else 0.0
        else:
            hitS = expit(-tau * (data[tt] - VEhat[tt, 0]))

        ES_t, VaR_t = VEhat[tt, 1], VEhat[tt, 0]
        if (not np.isfinite(ES_t)) or ES_t >= 0:
            return 1e7
        loss[tt] = -(1.0/alpha)/ES_t * hitS * (VaR_t - data[tt]) - (1.0/ES_t) * (ES_t - VaR_t) + np.log(-ES_t)

    Eloss = float(np.mean(loss))
    if not np.isfinite(Eloss):
        return 1e7

    # ===== Sanity checks / penalties (Matlab-style) =====
    # require ES < VaR; require both < 0; keep ES from going too extreme
    if np.any(VEhat[:, 0] < VEhat[:, 1]) or (np.nanmax(VEhat) > 0.0) or (np.nanmin(VEhat[:, 1]) < 5.0 * np.nanmin(data)):
        return 1e6

    # must see some hits; must be finite
    if np.any(~np.isfinite(VEhat)) or (np.sum(data <= VEhat[:, 0]) == 0):
        return 1e7

    return Eloss


def UGarchFit(y: np.ndarray,
              alpha: float,
              omega: float,
              initial_theta_raw: np.ndarray | None = None):
    """
    Matlab replication strategy for GARCH-FZ:
      1) Build theta0 using sample_VE, plus fixed beta0=0.95, gamma0=0.005
      2) Optimize smoothed hit objective with tau=5, then tau=20 (BFGS)
      3) Finalize with true objective tau=-1 using Nelder-Mead
    """
    y = np.asarray(y, dtype=float)
    if y.size < 10 or np.any(~np.isfinite(y)):
        raise ValueError("Bad training series.")

    if initial_theta_raw is None:
        vbar, ebar = sample_VE(y, alpha)  # both negative typically
        if (not np.isfinite(vbar)) or (not np.isfinite(ebar)) or (ebar >= 0) or (vbar >= 0):
            vbar, ebar = -1.0, -2.0

        c0 = vbar / ebar
        c0 = float(np.clip(c0, 1e-6, 1.0 - 1e-6))

        beta0 = 0.95
        gamma0 = 0.005
        initial_theta_raw = np.array([
            norm.ppf(beta0),
            np.log(gamma0),
            np.log(-ebar),     # since b=-exp(theta3) => theta3 = log(-b); take b0=ebar
            norm.ppf(c0),
        ], dtype=float)

    h0 = float(np.var(y)) if np.isfinite(np.var(y)) and np.var(y) > 0 else 1.0

    def obj(theta_raw, tau):
        return garch_FZ_LL_objective(theta_raw, y, alpha=alpha, tau=tau, omega=omega, h0=h0)

    # warm starts with smoothed hit
    # res1 = minimize(obj, initial_theta_raw, method="BFGS", args=(5.0,))
    # res2 = minimize(obj, res1.x,           method="BFGS", args=(20.0,))
    # finalize with true objective
    resf = minimize(obj, initial_theta_raw, method="Nelder-Mead", args=(-1.0,))
    return resf


def UGarchFor(y_hist: np.ndarray,
              y_oos: np.ndarray,
              omega: float,
              theta_raw_hat: np.ndarray,
              add_mean: float = 0.0):
    """
    Matlab-aligned forecasting:
      - variance recursion on demeaned series
      - VaR/ES = mean + [a,b]*sqrt(h_t)
    """
    y_hist = np.asarray(y_hist, dtype=float)
    y_oos  = np.asarray(y_oos,  dtype=float)

    beta, gamma, a, b, c = _transform_theta_garch_fz(theta_raw_hat)

    T_hist = y_hist.size
    T_oos  = y_oos.size
    if T_hist == 0:
        raise ValueError("Empty training series.")

    h = np.empty(T_hist + T_oos, dtype=float)
    h0 = float(np.var(y_hist))
    if (not np.isfinite(h0)) or h0 <= 0:
        h0 = 1.0
    h[0] = h0

    # replay history
    for t in range(1, T_hist):
        h[t] = omega + beta * h[t-1] + gamma * (y_hist[t-1] ** 2)
        if (not np.isfinite(h[t])) or h[t] <= 0:
            h[t] = 1.0

    v_hat = np.empty(T_oos, dtype=float)
    e_hat = np.empty(T_oos, dtype=float)

    for t in range(T_oos):
        prev_y = y_hist[-1] if t == 0 else y_oos[t-1]
        prev_h = h[T_hist-1] if t == 0 else h[T_hist + t - 1]
        h[T_hist + t] = omega + beta * prev_h + gamma * (prev_y ** 2)
        if (not np.isfinite(h[T_hist + t])) or h[T_hist + t] <= 0:
            h[T_hist + t] = 1.0

        s = np.sqrt(h[T_hist + t])
        v_hat[t] = add_mean + a * s
        e_hat[t] = add_mean + b * s

    return v_hat, e_hat


# =========================
# Driver
# =========================
def run_garch(df: pd.DataFrame,
              alpha: float = 0.05,
              results_dir: str | None = None,
              verbose: bool = True,
              min_train: int = 12) -> pd.DataFrame:
    """
    GARCH-FZ (Patton-Ziegel-Chen replication style) on panel data.

    Required columns: 'eom' (datetime), 'id', 'ret_exc_lead1m'
    Yearly re-fit: yr in 2014..2023, train on eom < yr-01-01, forecast within that year.
    """
    if "eom" not in df.columns:
        raise ValueError("Input df must contain 'eom'.")
    if "ret_exc_lead1m" not in df.columns or "id" not in df.columns:
        raise ValueError("Input df must contain 'ret_exc_lead1m' and 'id'.")

    dff = df.copy()
    dff["eom"] = pd.to_datetime(dff["eom"])
    dff = dff.sort_values(["id", "eom"]).reset_index(drop=True)

    all_preds = []
    ids = dff["id"].unique().tolist()

    id_iter = tqdm(ids, desc="GARCH-FZ per id", disable=not verbose)
    for sid in id_iter:
        sub = dff[dff["id"] == sid].copy().sort_values("eom")
        sub = sub[sub["eom"] < TEST_END]

        for yr in TEST_YEARS:
            cutoff = pd.Timestamp(f"{yr}-01-01")

            y_train = _prepare_training_series(sub, cutoff)
            y_train = y_train[np.isfinite(y_train)]
            if y_train.size < min_train:
                continue

            # demean for fitting (Matlab replication does this)
            mu = float(np.mean(y_train))
            y_train0 = y_train - mu

            oos_start = cutoff
            oos_end = min(pd.Timestamp(f"{yr+1}-01-01"), TEST_END)
            sub_oos = sub[(sub["eom"] >= oos_start) & (sub["eom"] < oos_end)]
            if sub_oos.empty:
                continue

            y_oos = sub_oos["ret_exc_lead1m"].to_numpy(dtype=float)
            ok = np.isfinite(y_oos)
            if not np.any(ok):
                continue
            # keep alignment by not dropping rows: forecast only for finite y, then reinsert
            y_oos0 = y_oos - mu

            # omega heuristic (since omega not identified in FZ GARCH; replication fixes omega externally)
            var_y = float(np.var(y_train0))
            omega = max(1e-8, 0.05 * var_y)

            try:
                opt = UGarchFit(y=y_train0, alpha=alpha, omega=omega, initial_theta_raw=None)
                theta_raw_hat = opt.x
            except Exception:
                continue

            # forecasts (keep NaN where y_oos is NaN)
            v_hat = np.full_like(y_oos, np.nan, dtype=float)
            e_hat = np.full_like(y_oos, np.nan, dtype=float)
            try:
                v_tmp, e_tmp = UGarchFor(
                    y_hist=y_train0,
                    y_oos=y_oos0[ok],
                    omega=omega,
                    theta_raw_hat=theta_raw_hat,
                    add_mean=mu,
                )
                v_hat[ok] = v_tmp
                e_hat[ok] = e_tmp
            except Exception:
                continue

            part = pd.DataFrame({
                "eom": sub_oos["eom"].to_numpy(),
                "id": sid,
                "y": y_oos,
                "v": v_hat,
                "e": e_hat,
                "model_year": yr,
            })
            all_preds.append(part)

    if not all_preds:
        return pd.DataFrame(columns=["eom", "id", "y", "v", "e", "model_year"])

    pred_df = pd.concat(all_preds, axis=0, ignore_index=True)
    pred_df = pred_df.sort_values(["id", "eom"]).reset_index(drop=True)

    if results_dir is not None:
        out_dir = os.path.join(results_dir, "GARCH")
        os.makedirs(out_dir, exist_ok=True)
        pred_df.to_csv(os.path.join(out_dir, "GARCH.csv"), index=False)

    return pred_df

__all__ = ["run_garch"]
