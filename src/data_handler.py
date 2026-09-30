"""
data_handler.py
---------------
Fetch, clean, and transform price data into return statistics ready for
Markowitz mean-variance optimisation.

Expected returns and covariances are estimated from daily *simple* returns.
Log returns are still returned for distribution plots; they are not the
inputs to the optimiser. A portfolio's simple return is the weighted sum
of asset simple returns. That is not true of log returns.

    mu  = mean(simple returns) × 252
    cov = sample covariance(simple returns) × 252     (divisor n − 1)
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


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns, P_t / P_{t-1} − 1."""
    return prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="any")


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns. Useful for distribution plots, not for portfolio weights."""
    return np.log(prices / prices.shift(1)).replace([np.inf, -np.inf], np.nan).dropna(how="any")


def annualise_stats(
    daily_returns: pd.DataFrame,
    shrink: bool = False,
) -> tuple[pd.Series, pd.DataFrame]:
    """Annualise the sample mean and a covariance of daily simple returns.

    The default covariance divisor is n − 1 (``DataFrame.cov``). Multiplying
    by 252 assumes returns are uncorrelated across days. ``shrink=True``
    replaces that covariance with the Ledoit-Wolf estimator (maximum-likelihood
    covariance, shrunk toward a scaled identity, then multiplied by 252).
    The mean is the sample mean either way.
    """
    if len(daily_returns) < 2:
        raise ValueError("Need at least two observations to estimate a covariance.")
    mu = daily_returns.mean() * TRADING_DAYS
    if shrink:
        from src.covariance import ledoit_wolf

        cov, _shrinkage = ledoit_wolf(daily_returns)
        cov = cov * TRADING_DAYS
    else:
        cov = daily_returns.cov() * TRADING_DAYS
    return mu, cov


def prepare_price_data(
    prices: pd.DataFrame,
    train_end: str | pd.Timestamp | None = None,
) -> dict:
    """Split prices into train and test without using the test window in mu or cov.

    The return dated t uses the close on t and the previous close, so it is
    known at the close on t. Training keeps every return on or before
    ``train_end``. The test set is every later session. A split date that is
    not itself a trading day therefore does not drop the next session.
    """
    prices = prices.sort_index()
    returns = compute_returns(prices)
    log_returns = compute_log_returns(prices)
    if len(returns) < 2:
        raise ValueError("Price history is too short to compute returns.")

    if train_end is None:
        split_idx = int(len(returns) * 0.80)
        train_end_ts = pd.Timestamp(returns.index[split_idx])
    else:
        train_end_ts = pd.Timestamp(train_end)

    train_returns = returns.loc[returns.index <= train_end_ts]
    test_returns = returns.loc[returns.index > train_end_ts]
    if len(train_returns) < 2:
        raise ValueError("Training window has fewer than two observations.")

    mu, cov = annualise_stats(train_returns)
    return {
        "prices": prices,
        "returns": returns,
        "log_returns": log_returns,
        "train_prices": prices.loc[prices.index <= train_end_ts],
        "test_prices": prices.loc[prices.index > train_end_ts],
        "train_returns": train_returns,
        "test_returns": test_returns,
        "mu": mu,
        "cov": cov,
        "tickers": list(prices.columns),
    }


def load_data(
    tickers: list[str],
    start: str,
    end: str,
    train_end: str | None = None,
    cache: bool = True,
) -> dict:
    """
    Download prices, compute simple returns, and split into train / test sets.

    Parameters
    ----------
    tickers   : list of ticker symbols
    start     : data start date  (ISO string)
    end       : data end date    (ISO string)
    train_end : last date whose return is in-sample.
                Everything after this date is the out-of-sample test set.
                Defaults to 80 % of the return observations.
    cache     : whether to cache the raw CSV

    Returns
    -------
    dict with keys:
        prices        – adjusted close prices
        returns       – full daily simple returns
        log_returns   – full daily log returns (exploration only)
        train_prices  – prices on or before the split
        test_prices   – prices strictly after the split
        train_returns – in-sample daily simple returns
        test_returns  – out-of-sample daily simple returns
        mu            – annualised mean simple returns (in-sample)
        cov           – annualised sample covariance (in-sample)
        tickers       – column order used everywhere else
    """
    prices = download_prices(tickers, start=start, end=end, cache=cache)
    return prepare_price_data(prices, train_end=train_end)


def simulate_random_portfolios(
    mu: np.ndarray | pd.Series,
    cov: np.ndarray | pd.DataFrame,
    n_portfolios: int = 5000,
    risk_free_rate: float = 0.04,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Monte-Carlo simulation of random long-only portfolios for visualisation.

    Draws are uniform on the simplex (Dirichlet with every parameter equal to 1).

    Returns
    -------
    pd.DataFrame with columns [ret, vol, sharpe, <asset columns…>]
    """
    rng = np.random.default_rng(seed)
    mu_arr = np.asarray(mu, dtype=float).ravel()
    cov_arr = np.asarray(cov, dtype=float)
    n = len(mu_arr)

    records = []
    for _ in range(n_portfolios):
        w = rng.dirichlet(np.ones(n))
        ret = float(w @ mu_arr)
        var = float(w @ cov_arr @ w)
        vol = float(np.sqrt(var)) if var > 0 else 0.0
        sharpe = (ret - risk_free_rate) / vol if vol > 0 else np.nan
        records.append([ret, vol, sharpe, *w])

    tickers = list(mu.index) if hasattr(mu, "index") else [f"w_{i}" for i in range(n)]
    cols = ["ret", "vol", "sharpe"] + tickers
    return pd.DataFrame(records, columns=cols)
