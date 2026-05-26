from __future__ import annotations

import numpy as np
import pandas as pd


def fz0_loss(y, v, e, alpha: float = 0.05) -> np.ndarray:
    """Compute the Fissler-Ziegel FZ0 loss used for VaR/ES forecasts."""

    y_arr = np.asarray(y, dtype=float)
    v_arr = np.asarray(v, dtype=float)
    e_arr = np.asarray(e, dtype=float)
    indicator = (y_arr <= v_arr).astype(float)
    term1 = -(1.0 / (alpha * e_arr)) * indicator * (v_arr - y_arr)
    term2 = (v_arr / e_arr) + np.log(-100.0 * e_arr) - 1.0
    return term1 + term2


def add_fz0_loss(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Return a copy of ``df`` with a consistently recomputed ``loss`` column."""

    out = df.copy()
    out["loss"] = fz0_loss(out["y"], out["v"], out["e"], alpha=alpha)
    return out
