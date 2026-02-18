"""
data_handler.py
---------------
Fetch, clean, and transform price data into return statistics ready for
Markowitz mean-variance optimisation.

Key outputs
-----------
  mu  : annualised expected returns  (pd.Series,  shape n)
  cov : annualised covariance matrix (pd.DataFrame, shape n×n)
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from data.fetch_data import download_prices


# ── Constants ─────────────────────────────────────────────────────────────────
TRADING_DAYS = 252          # annualisation factor


# ── Public API ────────────────────────────────────────────────────────────────

def load_data(
    tickers: list[str],
    start: str,
    end: str,
    train_end: str | None = None,
    cache: bool = True,
) -> dict:
    """
    Download prices, compute log-returns, and split into train / test sets.

    Parameters
    ----------
    tickers   : list of ticker symbols
    start     : data start date  (ISO string)
    end       : data end date    (ISO string)
    train_end : last date of in-sample (train) window.
                Everything after this date is the out-of-sample test set.
                Defaults to 80 % of the date range.
    cache     : whether to cache the raw CSV

    Returns
    -------
    dict with keys:
        prices        – full adjusted-close DataFrame
        log_returns   – full daily log-return DataFrame
        train_prices  – in-sample prices
        test_prices   – out-of-sample prices
        train_returns – in-sample daily log-returns
        test_returns  – out-of-sample daily log-returns
        mu            – annualised mean returns (in-sample)
        cov           – annualised covariance matrix (in-sample)
        tickers       – list of tickers
    """
    prices = download_prices(tickers, start=start, end=end, cache=cache)
    log_returns = np.log(prices / prices.shift(1)).dropna()

    # ── Train / test split ────────────────────────────────────────────────────
    if train_end is None:
        split_idx = int(len(log_returns) * 0.80)
        train_end = log_returns.index[split_idx].strftime("%Y-%m-%d")

    train_returns = log_returns.loc[:train_end]
    test_returns  = log_returns.loc[train_end:].iloc[1:]   # exclude boundary row

    train_prices  = prices.loc[:train_end]
    test_prices   = prices.loc[train_end:]

    # ── Annualised statistics (in-sample) ─────────────────────────────────────
    mu  = train_returns.mean() * TRADING_DAYS
    cov = train_returns.cov()  * TRADING_DAYS

    return {
        "prices":        prices,
        "log_returns":   log_returns,
        "train_prices":  train_prices,
        "test_prices":   test_prices,
        "train_returns": train_returns,
        "test_returns":  test_returns,
        "mu":            mu,
        "cov":           cov,
        "tickers":       list(prices.columns),
    }


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns from a price DataFrame."""
    return np.log(prices / prices.shift(1)).dropna()


def annualise_stats(daily_returns: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """
    Given daily log-return DataFrame return (mu, cov) annualised.

    Returns
    -------
    mu  : pd.Series    – annualised expected returns per asset
    cov : pd.DataFrame – annualised covariance matrix
    """
    mu  = daily_returns.mean() * TRADING_DAYS
    cov = daily_returns.cov()  * TRADING_DAYS
    return mu, cov


def simulate_random_portfolios(
    mu: np.ndarray | pd.Series,
    cov: np.ndarray | pd.DataFrame,
    n_portfolios: int = 5000,
    risk_free_rate: float = 0.04,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Monte-Carlo simulation of random long-only portfolios for visualisation.

    Parameters
    ----------
    mu              : annualised expected returns (length n)
    cov             : annualised covariance matrix (n×n)
    n_portfolios    : number of random portfolios to simulate
    risk_free_rate  : used to compute Sharpe ratio
    seed            : random seed for reproducibility

    Returns
    -------
    pd.DataFrame with columns [ret, vol, sharpe, w_0, w_1, …, w_n]
    """
    rng = np.random.default_rng(seed)
    mu_arr  = np.asarray(mu)
    cov_arr = np.asarray(cov)
    n = len(mu_arr)

    records = []
    for _ in range(n_portfolios):
        raw = rng.exponential(1, size=n)       # Dirichlet-like draw
        w   = raw / raw.sum()                  # sums to 1, all >= 0
        ret = w @ mu_arr
        vol = np.sqrt(w @ cov_arr @ w)
        sharpe = (ret - risk_free_rate) / vol
        records.append([ret, vol, sharpe, *w])

    tickers = list(mu.index) if hasattr(mu, "index") else [f"w_{i}" for i in range(n)]
    cols = ["ret", "vol", "sharpe"] + tickers
    return pd.DataFrame(records, columns=cols)
