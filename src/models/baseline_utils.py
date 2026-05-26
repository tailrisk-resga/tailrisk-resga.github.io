"""Shared utilities for GAS and GARCH baselines."""

import numpy as np
import pandas as pd
from scipy.special import expit

TEST_START = pd.Timestamp('2014-01-01')
TEST_END = pd.Timestamp('2024-01-01')   # exclusive
TEST_YEARS = list(range(2014, 2024))     # 2014-2023 inclusive -> 10 annual models

def L_FZ0(y:np.array,
          v:np.array,
          e:np.array,
          alpha:float):
    """FZ Loss Function
    
    Args:
        y (np.array): true value
        v (np.array): estimated quantile
        e (np.array): estimated expected shortfall
        alpha (float): quantile level

    Returns:
        loss: loss value
    """    
    indicator = (y <= v).astype(float)
    term1 = - (1 / (alpha * e)) * indicator * (v - y)
    term2 = (v / e) + np.log(-100 * e) - 1
    loss = term1 + term2
    return np.mean(loss)

def L_FZ0_smooth(y:np.array,
                 v:np.array,
                 e:np.array,
                 alpha:float,
                 smoothing:float):
    """smoothed FZ Loss Function (only used for para initial)

    Args:
        y (np.array): true value
        v (np.array): estimated quantile
        e (np.array): estimated expected shortfall
        alpha (float): quantile level
        smoothing (float): hyperparameter, controling the degree of smoothing

    Returns:
        loss: loss value
    """
        
    indicator_smooth = expit(-smoothing * (y - v))
    term1 = - (1 / (alpha * e)) * indicator_smooth * (v - y)
    term2 = (v / e) + np.log(-e) - 1
    
    return term1 + term2

def sample_VE(y: np.ndarray, alpha: float = 0.05):
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return np.nan, np.nan
    v = np.quantile(y, alpha)
    tail = y[y <= v]
    if tail.size == 0:
        return float(v), float(v * 1.1)
    e = tail.mean()
    if e > v:
        e = v
    return float(v), float(e)

def _prepare_training_series(df_stock: pd.DataFrame, cutoff: pd.Timestamp) -> np.ndarray:
    """Return y series strictly before cutoff."""
    use = df_stock.loc[df_stock["eom"] < cutoff].sort_values("eom")
    return use["ret_exc_lead1m"].astype(float).to_numpy()

__all__ = [
    "TEST_START",
    "TEST_END",
    "TEST_YEARS",
    "L_FZ0",
    "L_FZ0_smooth",
    "sample_VE",
    "_prepare_training_series",
]
