"""
backtest.py
-----------
Out-of-sample evaluation helpers.

Static weights in the app are applied to simple returns, which is daily
rebalancing back to those weights (no costs). ``walk_forward_max_sharpe``
refits at the first session of each out-of-sample month using only returns
strictly before that session, then holds the new weights until the next refit.
"""

from __future__ import annotations

import pandas as pd

from src.data_handler import annualise_stats
from src.metrics import portfolio_daily_returns
from src.optimizer import max_sharpe


def _month_starts(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """First timestamp of each calendar month present in ``index``."""
    starts: list[pd.Timestamp] = []
    seen: set[pd.Period] = set()
    for ts, period in zip(index, index.to_period("M"), strict=True):
        if period not in seen:
            seen.add(period)
            starts.append(pd.Timestamp(ts))
    return starts


def walk_forward_max_sharpe(
    returns: pd.DataFrame,
    oos_start,
    risk_free_rate: float = 0.04,
    max_weight: float | None = None,
    min_history: int = 60,
) -> pd.Series:
    """Monthly refit of the long-only maximum-Sharpe portfolio.

    For a rebalance session t, the fit uses ``returns`` with index < t, and
    those weights are applied to returns on [t, next rebalance). A month that
    cannot be solved is omitted rather than filled with a later fit.
    """
    if returns.empty:
        return pd.Series(dtype=float)
    oos = returns.loc[returns.index > pd.Timestamp(oos_start)]
    if oos.empty:
        return pd.Series(dtype=float)

    starts = _month_starts(oos.index)
    bounds = starts + [pd.Timestamp(oos.index[-1]) + pd.Timedelta(days=1)]
    pieces: list[pd.Series] = []
    for start, end in zip(bounds[:-1], bounds[1:], strict=True):
        history = returns.loc[returns.index < start]
        window = returns.loc[(returns.index >= start) & (returns.index < end)]
        if len(history) < min_history or window.empty:
            continue
        mu, cov = annualise_stats(history)
        fitted = max_sharpe(mu, cov, risk_free_rate=risk_free_rate, max_weight=max_weight)
        if fitted["weights"] is None:
            continue
        pieces.append(portfolio_daily_returns(fitted["weights"], window))
    if not pieces:
        return pd.Series(dtype=float)
    out = pd.concat(pieces)
    out = out[~out.index.duplicated(keep="first")]
    out.name = "MSR (monthly rebalance)"
    return out
