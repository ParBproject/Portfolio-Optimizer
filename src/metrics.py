"""
metrics.py
----------
Performance and risk metrics for portfolio evaluation.

Covers
------
  - Annualised return, volatility, Sharpe ratio
  - Maximum drawdown
  - Calmar ratio
  - Portfolio cumulative returns from weights + daily returns
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ── Core statistics ────────────────────────────────────────────────────────────

def annualised_return(daily_returns: pd.Series | np.ndarray) -> float:
    """
    Compound Annualised Growth Rate (CAGR) from daily log-returns.

    CAGR = exp( mean(r) × T ) - 1  where T = TRADING_DAYS
    """
    r = np.asarray(daily_returns)
    return float(np.exp(r.mean() * TRADING_DAYS) - 1)


def annualised_volatility(daily_returns: pd.Series | np.ndarray) -> float:
    """Annualised volatility (std dev of daily log-returns × √252)."""
    r = np.asarray(daily_returns)
    return float(r.std() * np.sqrt(TRADING_DAYS))


def sharpe_ratio(
    daily_returns: pd.Series | np.ndarray,
    risk_free_rate: float = 0.04,
) -> float:
    """
    Sharpe ratio = (annualised_return - rf) / annualised_volatility.

    Parameters
    ----------
    daily_returns   : daily log-returns
    risk_free_rate  : annualised risk-free rate (default 4 %)
    """
    ret = annualised_return(daily_returns)
    vol = annualised_volatility(daily_returns)
    return (ret - risk_free_rate) / vol if vol > 0 else np.nan


def max_drawdown(cumulative_returns: pd.Series | np.ndarray) -> float:
    """
    Maximum drawdown from a cumulative-return (or price) series.

    MDD = max over time of  (peak - trough) / peak

    Parameters
    ----------
    cumulative_returns : cumulative wealth index (starts at 1 or any base)

    Returns
    -------
    float   Maximum drawdown as a *positive* fraction (e.g. 0.30 = 30 % loss).
    """
    cum = np.asarray(cumulative_returns, dtype=float)
    running_max = np.maximum.accumulate(cum)
    drawdowns   = (running_max - cum) / running_max
    return float(drawdowns.max())


def calmar_ratio(
    daily_returns: pd.Series | np.ndarray,
    risk_free_rate: float = 0.04,
) -> float:
    """Calmar = annualised excess return / max drawdown."""
    cum = np.exp(np.cumsum(np.asarray(daily_returns)))
    mdd = max_drawdown(cum)
    ret = annualised_return(daily_returns)
    return (ret - risk_free_rate) / mdd if mdd > 0 else np.nan


# ── Portfolio-level helpers ────────────────────────────────────────────────────

def portfolio_daily_returns(
    weights: np.ndarray | pd.Series,
    asset_returns: pd.DataFrame,
) -> pd.Series:
    """
    Compute daily portfolio log-returns given fixed weights.

    Note: w @ log_returns is an approximation for daily rebalancing.
    For buy-and-hold, use simple returns (exp(log_return)-1).

    Parameters
    ----------
    weights       : portfolio weights (n,), must sum to ~1
    asset_returns : daily log-return DataFrame (T × n)

    Returns
    -------
    pd.Series  daily portfolio log-returns
    """
    w = np.asarray(weights)
    w = w / w.sum()   # normalise
    return asset_returns.values @ w


def cumulative_wealth(daily_log_returns: np.ndarray | pd.Series) -> pd.Series:
    """Convert daily log-returns to cumulative wealth index (starts at 1)."""
    r = np.asarray(daily_log_returns)
    index = daily_log_returns.index if hasattr(daily_log_returns, "index") else range(len(r))
    return pd.Series(np.exp(np.cumsum(r)), index=index)


# ── Summary report ─────────────────────────────────────────────────────────────

def performance_summary(
    portfolio_returns: pd.Series,
    label: str = "Portfolio",
    risk_free_rate: float = 0.04,
) -> pd.Series:
    """
    One-row summary of key performance metrics.

    Returns
    -------
    pd.Series with index: [Annualised Return, Annualised Vol, Sharpe Ratio,
                            Max Drawdown, Calmar Ratio]
    """
    cum = cumulative_wealth(portfolio_returns)
    return pd.Series(
        {
            "Annualised Return": annualised_return(portfolio_returns),
            "Annualised Vol":    annualised_volatility(portfolio_returns),
            "Sharpe Ratio":      sharpe_ratio(portfolio_returns, risk_free_rate),
            "Max Drawdown":      max_drawdown(cum),
            "Calmar Ratio":      calmar_ratio(portfolio_returns, risk_free_rate),
        },
        name=label,
    )


def compare_portfolios(
    portfolios: dict[str, pd.Series],
    risk_free_rate: float = 0.04,
) -> pd.DataFrame:
    """
    Build a comparison table for multiple portfolio daily-return series.

    Parameters
    ----------
    portfolios : {label: daily_log_returns_series}

    Returns
    -------
    pd.DataFrame  rows = metrics, columns = portfolio labels
    """
    rows = [
        performance_summary(ret, label=label, risk_free_rate=risk_free_rate)
        for label, ret in portfolios.items()
    ]
    df = pd.concat(rows, axis=1)
    fmt = {
        "Annualised Return": "{:.2%}",
        "Annualised Vol":    "{:.2%}",
        "Sharpe Ratio":      "{:.3f}",
        "Max Drawdown":      "{:.2%}",
        "Calmar Ratio":      "{:.3f}",
    }
    return df, fmt
