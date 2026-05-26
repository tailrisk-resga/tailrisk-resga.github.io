"""GAS baseline."""

import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm
from tqdm import tqdm

from models.baseline_utils import sample_VE

###############################################################################
# === One-factor GAS (score-driven) model for joint VaR/ES, FZ0 loss ===
# Reference: Patton et al. (2019), Section 2.3, score-driven VaR/ES modeling.
###############################################################################


# -----------------------------
# (A) Matlab-aligned transform
# -----------------------------


def _transform_theta(theta_raw: np.ndarray):
    """
    Matlab GAS_onefactor_LL3:
      beta = normcdf(theta(1))
      gamma = exp(theta(2))
      b = -exp(theta(3))
      c = normcdf(theta(4))
      a = c*b
      omega = 0
      kalpha = 1
    """
    beta  = norm.cdf(theta_raw[0])          # (0,1)
    gamma = np.exp(theta_raw[1])            # >0
    b     = -np.exp(theta_raw[2])           # <0
    c     = norm.cdf(theta_raw[3])          # (0,1)
    a     = c * b                           # a in (b,0), ensures ES<VaR<0
    omega = 0.0
    kalpha = 1.0
    return omega, beta, gamma, a, b, c, kalpha


# -----------------------------------------
# (B) Matlab-aligned recursion + loss + checks
# -----------------------------------------
def gas_onefactor_LL3_objective(theta_raw: np.ndarray,
                                Y: np.ndarray,
                                alpha: float = 0.05,
                                tau: float = -1.0,
                                return_path: bool = False):
    """
    Mirrors GAS_onefactor_LL3.m:
      fhat(1)=omega/(1-beta), VEhat(1,:)=[a,b]*exp(fhat(1))
      hitS = I(Y(1)<VaR(1)) or logistic smoothing
      loss(t) = -1/alpha/ES*hit*(VaR-Y) -1/ES*(ES-VaR) + log(-ES)

      fhat(tt) = omega + beta*fhat(tt-1) + gamma/kalpha/ES(tt-1) * (1/alpha*hitSL*Y(tt-1) - ES(tt-1))
      VEhat(tt,:) = [a,b]*exp(fhat(tt))
      hitS updated using Y(tt) and VaR(tt)

    Penalties:
      if any(VaR < ES) or max(VEhat) >= 0 => 1e6
      if nan/inf/complex or hits==0 => 1e7
    """
    omega, beta, gamma, a, b, c, kalpha = _transform_theta(theta_raw)

    Y = np.asarray(Y, dtype=float)
    T = len(Y)
    if T < 2:
        val = 1e7
        return (val, None) if return_path else val

    fhat = np.full(T, np.nan, dtype=float)
    VEhat = np.full((T, 2), np.nan, dtype=float)  # col0 VaR, col1 ES
    loss = np.zeros(T, dtype=float)

    # Matlab: fhat(1) = omega/(1-beta). Here omega=0 => 0, but keep formula
    fhat[0] = omega / max(1e-12, (1.0 - beta))
    VEhat[0, :] = np.array([a, b]) * np.exp(fhat[0])

    # hitS at t=0 uses Y(0) and VaR(0), strict "<" as Matlab
    if tau == -1:
        hitS = 1.0 if (Y[0] < VEhat[0, 0]) else 0.0
    else:
        # Matlab: 1/(1+exp(tau*(Y-VaR)))
        hitS = expit(-tau * (Y[0] - VEhat[0, 0]))

    # loss at t=0
    ES0 = VEhat[0, 1]
    VaR0 = VEhat[0, 0]
    

    loss[0] = -(1.0/alpha)/ES0 * hitS * (VaR0 - Y[0]) \
              - (1.0/ES0) * (ES0 - VaR0) \
              + np.log(-ES0)

    # recursion
    for tt in range(1, T):
        hitSL = hitS  # lagged hit

        ES_prev = VEhat[tt-1, 1]

        fhat[tt] = omega + beta * fhat[tt-1] + (gamma / kalpha) / ES_prev * ((1.0/alpha) * hitSL * Y[tt-1] - ES_prev)
        VEhat[tt, :] = np.array([a, b]) * np.exp(fhat[tt])

        # update hitS using current Y(tt) and current VaR(tt), strict "<"
        if tau == -1:
            hitS = 1.0 if (Y[tt] < VEhat[tt, 0]) else 0.0
        else:
            hitS = expit(-tau * (Y[tt] - VEhat[tt, 0]))

        ES_t = VEhat[tt, 1]
        VaR_t = VEhat[tt, 0]

        loss[tt] = -(1.0/alpha)/ES_t * hitS * (VaR_t - Y[tt]) \
                   - (1.0/ES_t) * (ES_t - VaR_t) \
                   + np.log(-ES_t)

    # ---- Matlab penalties / sanity checks ----
    Eloss = float(np.mean(loss))

    # require ES < VaR, and both < 0 (Matlab checks max(max(VEhat))>=0)
    if np.any(VEhat[:, 0] < VEhat[:, 1]) or (np.nanmax(VEhat) >= 0.0):
        Eloss = 1e6

    # require finite, real, and at least some hits
    hits = int(np.sum(Y < VEhat[:, 0]))  # Matlab uses strict "<" in hit, but check uses <=; keep strict consistent
    if (not np.isfinite(Eloss)) or np.any(~np.isfinite(VEhat)) or hits == 0:
        Eloss = 1e7

    if return_path:
        # Matlab also outputs factor series; we can attach fhat as third column if needed
        return Eloss, np.column_stack([VEhat, fhat]), loss
    return Eloss


# -----------------------------
# (C) Matlab-aligned fit routine
# -----------------------------
def GASFit_LL3(y: np.ndarray,
               alpha: float = 0.05,
               init_theta_raw: np.ndarray = None):
    """
    Matlab typical strategy: optimize smoothed objective first (tau=5,20),
    then true objective (tau=-1) with Nelder-Mead.
    """
    y = np.asarray(y, dtype=float)
    if init_theta_raw is None:
        vbar, ebar = sample_VE(y, alpha)   # vbar<0, ebar<0
        c0 = vbar / ebar                # in (0,1)
        c0 = float(np.clip(c0, 1e-6, 1 - 1e-6))
        init_theta_raw = np.array([
            norm.ppf(0.95),       # beta0=0.95  -> theta1
            np.log(0.005),        # gamma0=0.005 -> theta2
            np.log(-ebar),        # b0=ebar      -> theta3 (since b=-exp(theta3))
            norm.ppf(c0),         # c0=vbar/ebar -> theta4
        ], dtype=float)
    

    def obj_tau(theta_raw, tau):
        return gas_onefactor_LL3_objective(theta_raw, y, alpha=alpha, tau=tau, return_path=False)

    # warm with smoothing in hit (as Matlab allows via tau)
    # res1 = minimize(obj_tau, init_theta_raw, method="BFGS", args=(5.0,))
    # res2 = minimize(obj_tau, res1.x, method="BFGS", args=(20.0,))
    resf = minimize(obj_tau, init_theta_raw, method="Nelder-Mead", args=(-1.0,))
    return resf


# -----------------------------
# (D) Matlab-aligned one-step-ahead forecasting (yearly refit)
# -----------------------------
def GASFor_LL3(y_hist: np.ndarray, y_oos: np.ndarray, alpha: float, theta_raw_hat: np.ndarray):
    """
    Produce OOS forecasts using fitted theta_raw, matching the same recursion.
    Forecast for OOS month t uses last available realized y (t-1) to update f, then compute VaR/ES.
    """
    y_hist = np.asarray(y_hist, dtype=float)
    y_oos = np.asarray(y_oos, dtype=float)
    omega, beta, gamma, a, b, c, kalpha = _transform_theta(theta_raw_hat)

    # replay history with tau=-1 to obtain last f and last hit
    Eloss, VE_path, loss_path = gas_onefactor_LL3_objective(theta_raw_hat, y_hist, alpha=alpha, tau=-1, return_path=True)
    # VE_path columns: [VaR, ES, f]
    f_prev = VE_path[-1, 2]
    VaR_prev = VE_path[-1, 0]

    # hit based on last in-sample obs, strict "<"
    hit_prev = 1.0 if (y_hist[-1] < VaR_prev) else 0.0
    ES_prev = VE_path[-1, 1]

    # first update to get state used for first OOS forecast (uses y_hist[-1], hit_prev, ES_prev)
    f_prev = omega + beta * f_prev + (gamma / kalpha) / ES_prev * ((1.0/alpha) * hit_prev * y_hist[-1] - ES_prev)

    T_oos = len(y_oos)
    v_hat = np.zeros(T_oos, dtype=float)
    e_hat = np.zeros(T_oos, dtype=float)

    for t in range(T_oos):
        exp_f = np.exp(f_prev)
        v_hat[t] = a * exp_f
        e_hat[t] = b * exp_f

        # update with realized y_oos[t] for next step
        hit = 1.0 if (y_oos[t] < v_hat[t]) else 0.0
        ES_t = e_hat[t]
        f_prev = omega + beta * f_prev + (gamma / kalpha) / ES_t * ((1.0/alpha) * hit * y_oos[t] - ES_t)

    return v_hat, e_hat


def run_gas(df: pd.DataFrame,
            alpha: float = 0.05,
            results_dir: str = None,
            verbose: bool = True) -> pd.DataFrame:
    if 'eom' not in df.columns:
        raise ValueError("Input df must contain an 'eom' column.")
    if 'ret_exc_lead1m' not in df.columns or 'id' not in df.columns:
        raise ValueError("Input df must contain columns 'ret_exc_lead1m' and 'id'.")

    TEST_END = pd.Timestamp('2024-01-01')
    TEST_YEARS = list(range(2014, 2024))

    dff = df.copy()
    dff['eom'] = pd.to_datetime(dff['eom'])
    dff = dff.sort_values(['id', 'eom']).reset_index(drop=True)

    all_preds = []
    ids = dff['id'].unique().tolist()
    it = tqdm(ids, desc="GAS per id", disable=not verbose)

    for sid in it:
        sub = dff[dff['id'] == sid].copy().sort_values('eom')
        sub = sub[sub['eom'] < TEST_END]


        for yr in TEST_YEARS:
            cutoff = pd.Timestamp(f'{yr}-01-01')
            y_train = sub.loc[sub['eom'] < cutoff, 'ret_exc_lead1m'].astype(float).to_numpy()

            if y_train.size < 12:
                continue

            oos_start = cutoff
            oos_end = min(pd.Timestamp(f'{yr+1}-01-01'), TEST_END)
            sub_oos = sub[(sub['eom'] >= oos_start) & (sub['eom'] < oos_end)]
            if sub_oos.empty:
                continue
            y_oos = sub_oos['ret_exc_lead1m'].to_numpy(dtype=float)

            
            try:
                opt = GASFit_LL3(y_train, alpha=alpha)
                theta_raw_hat = opt.x
            except Exception:
                continue

            # forecasts
            try:
                v_hat, e_hat = GASFor_LL3(y_hist=y_train, y_oos=y_oos, alpha=alpha, theta_raw_hat=theta_raw_hat)
            except Exception:
                continue

            part = pd.DataFrame({
                'eom': sub_oos['eom'].to_numpy(),
                'id': sid,
                'y': y_oos,
                'v': v_hat,
                'e': e_hat,
                'model_year': yr,
            })
            all_preds.append(part)

    if not all_preds:
        return pd.DataFrame(columns=['eom', 'id', 'y', 'v', 'e', 'model_year'])

    pred_df = pd.concat(all_preds, axis=0, ignore_index=True).sort_values(['id', 'eom']).reset_index(drop=True)

    if results_dir is not None:
        out_dir = os.path.join(results_dir, 'GAS')
        os.makedirs(out_dir, exist_ok=True)
        pred_df.to_csv(os.path.join(out_dir, 'GAS.csv'), index=False)

    return pred_df

__all__ = ["run_gas"]
