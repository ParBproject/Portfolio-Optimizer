"""
metrics.py
----------
Performance and risk metrics for portfolio evaluation.

Inputs are daily *simple* returns (P_t / P_{t-1} - 1), not log returns.
A long-only portfolio that is rebalanced to fixed weights each day has
simple return wᵀ r. That identity does not hold for log returns.

Annualisation uses 252 trading days and assumes daily returns are
uncorrelated through time. The Sharpe ratio uses the arithmetic mean
excess return. CAGR is reported separately and is the numerator of the
Calmar ratio.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
_WEIGHT_SUM_ATOL = 1e-4


def _as_returns(daily_returns: pd.Series | np.ndarray) -> np.ndarray:
    if isinstance(daily_returns, pd.Series):
        values = daily_returns.to_numpy(dtype=float)
    else:
        values = np.asarray(daily_returns, dtype=float)
    return np.ravel(values)


def annualised_return(daily_returns: pd.Series | np.ndarray) -> float:
    """Arithmetic annualised return: mean(daily simple return) × 252.

    This is the expected-return convention used by the mean-variance
    optimiser and by the Sharpe ratio. For compound growth, use ``cagr``.
    """
    r = _as_returns(daily_returns)
    if r.size == 0 or not np.all(np.isfinite(r)):
        return np.nan
    return float(np.mean(r) * TRADING_DAYS)


def cagr(daily_returns: pd.Series | np.ndarray) -> float:
    """Compound annual growth rate from daily simple returns.

    CAGR = (Π (1 + r_t))^(252 / T) − 1.
    """
    r = _as_returns(daily_returns)
    if r.size == 0 or not np.all(np.isfinite(r)):
        return np.nan
    growth = 1.0 + r
    if np.any(growth <= 0):
        return np.nan
    wealth = float(np.prod(growth))
    return wealth ** (TRADING_DAYS / r.size) - 1.0


def annualised_volatility(daily_returns: pd.Series | np.ndarray) -> float:
    """Sample standard deviation of daily simple returns, times √252.

    The sample divisor is T − 1, matching ``DataFrame.cov``, so a
    one-asset portfolio's volatility equals the square root of its
    annualised variance.
    """
    r = _as_returns(daily_returns)
    if r.size < 2 or not np.all(np.isfinite(r)):
        return np.nan
    return float(np.std(r, ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(
    daily_returns: pd.Series | np.ndarray,
    risk_free_rate: float = 0.04,
) -> float:
    """Sharpe ratio from daily simple returns.

    (mean(r) × 252 − rf) / (std(r, ddof=1) × √252)

    ``risk_free_rate`` is an annualised, constant rate. The numerator is
    the arithmetic excess return, not CAGR minus the risk-free rate.
    """
    r = _as_returns(daily_returns)
    vol = annualised_volatility(r)
    if r.size == 0 or not np.isfinite(vol) or vol <= 0:
        return np.nan
    return (float(np.mean(r) * TRADING_DAYS) - risk_free_rate) / vol


def max_drawdown(cumulative_returns: pd.Series | np.ndarray) -> float:
    """Maximum drawdown of a wealth index, as a positive fraction.

    The series must include the starting net asset value. ``cumulative_wealth``
    starts at 1, so a loss on the first day is counted.
    """
    cum = np.asarray(cumulative_returns, dtype=float).ravel()
    cum = cum[np.isfinite(cum)]
    if cum.size == 0 or np.any(cum < 0):
        return np.nan
    running_max = np.maximum.accumulate(cum)
    valid = running_max > 0
    if not np.any(valid):
        return np.nan
    drawdowns = np.zeros_like(cum)
    drawdowns[valid] = (running_max[valid] - cum[valid]) / running_max[valid]
    return float(drawdowns.max())


def calmar_ratio(daily_returns: pd.Series | np.ndarray) -> float:
    """Calmar ratio = CAGR / maximum drawdown.

    Young (1991) does not subtract a risk-free rate.
    """
    wealth = cumulative_wealth(daily_returns)
    drawdown = max_drawdown(wealth)
    growth = cagr(daily_returns)
    if not np.isfinite(drawdown) or drawdown <= 0 or not np.isfinite(growth):
        return np.nan
    return growth / drawdown


def portfolio_daily_returns(
    weights: np.ndarray | pd.Series,
    asset_returns: pd.DataFrame,
) -> pd.Series:
    """Daily simple portfolio returns for fixed weights, rebalanced every day.

    Portfolio return on day t is wᵀ r_t. Labelled weights are aligned to
    ``asset_returns`` columns. Weights must already sum to 1 within
    ``1e-4``; a small residual is rescaled, a large one is rejected.
    """
    if not isinstance(asset_returns, pd.DataFrame):
        raise TypeError("asset_returns must be a DataFrame of simple returns")
    if isinstance(weights, pd.Series):
        aligned = weights.reindex(asset_returns.columns)
        if aligned.isna().any():
            missing = [str(col) for col in asset_returns.columns[aligned.isna()]]
            raise ValueError(f"weights are missing assets: {missing}")
        w = aligned.to_numpy(dtype=float)
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.size != asset_returns.shape[1]:
            raise ValueError(
                f"weights length {w.size} does not match {asset_returns.shape[1]} assets"
            )
    if not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite")
    total = float(w.sum())
    if abs(total - 1.0) > _WEIGHT_SUM_ATOL:
        raise ValueError(f"weights must sum to 1 (got {total:.6f})")
    w = w / total
    simple = asset_returns.to_numpy(dtype=float) @ w
    return pd.Series(simple, index=asset_returns.index, name="portfolio")


def cumulative_wealth(daily_returns: np.ndarray | pd.Series) -> pd.Series:
    """Wealth index from daily simple returns, starting at 1 before the first day."""
    if isinstance(daily_returns, pd.Series):
        r = daily_returns.astype(float)
    else:
        r = pd.Series(np.asarray(daily_returns, dtype=float).ravel())
    if r.empty:
        return pd.Series(dtype=float)
    wealth = (1.0 + r).cumprod()
    if isinstance(r.index, pd.DatetimeIndex):
        start_ts = pd.Timestamp(r.index[0]) - pd.tseries.offsets.BDay(1)
        if start_ts in wealth.index:
            start_ts = pd.Timestamp(r.index[0]) - pd.Timedelta(days=1)
        starter = pd.Series([1.0], index=pd.DatetimeIndex([start_ts]), name=r.name)
    else:
        starter = pd.Series([1.0], index=[r.index[0] - 1], name=r.name)
    return pd.concat([starter, wealth])


def performance_summary(
    portfolio_returns: pd.Series,
    label: str = "Portfolio",
    risk_free_rate: float = 0.04,
) -> pd.Series:
    """One-column summary. Sharpe uses the arithmetic return, Calmar uses CAGR."""
    return pd.Series(
        {
            "Arithmetic Return": annualised_return(portfolio_returns),
            "CAGR": cagr(portfolio_returns),
            "Annualised Vol": annualised_volatility(portfolio_returns),
            "Sharpe Ratio": sharpe_ratio(portfolio_returns, risk_free_rate),
            "Max Drawdown": max_drawdown(cumulative_wealth(portfolio_returns)),
            "Calmar Ratio": calmar_ratio(portfolio_returns),
        },
        name=label,
    )


def compare_portfolios(
    portfolios: dict[str, pd.Series],
    risk_free_rate: float = 0.04,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Comparison table (metrics × portfolios) and a display format map."""
    rows = [
        performance_summary(ret, label=label, risk_free_rate=risk_free_rate)
        for label, ret in portfolios.items()
    ]
    df = pd.concat(rows, axis=1)
    fmt = {
        "Arithmetic Return": "{:.2%}",
        "CAGR": "{:.2%}",
        "Annualised Vol": "{:.2%}",
        "Sharpe Ratio": "{:.3f}",
        "Max Drawdown": "{:.2%}",
        "Calmar Ratio": "{:.3f}",
    }
    return df, fmt


def format_comparison(table: pd.DataFrame, formats: dict[str, str]) -> pd.DataFrame:
    """Format a comparison table without pandas Styler.

    Styler needs jinja2 >= 3.1.5. That package is not a dependency, and the
    copy of jinja2 on a typical system Python is older, so ``DataFrame.style``
    raises and the backtest tab dies after the frontier has already been drawn.
    """
    display = pd.DataFrame(index=table.index, columns=table.columns, dtype=object)
    for metric in table.index:
        template = formats.get(str(metric))
        for column in table.columns:
            value = table.loc[metric, column]
            if template is None or not np.isfinite(value):
                display.loc[metric, column] = "" if not np.isfinite(value) else str(value)
            else:
                display.loc[metric, column] = template.format(value)
    return display
