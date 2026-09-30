"""
backtest.py
-----------
Out-of-sample evaluation.

Static weights in the app are a monthly constant-mix: trade back to the
target on the first session of each month, let the weights drift on the
days in between, and charge a proportional cost on the dollars traded.
``walk_forward_max_sharpe`` refits at each of those sessions using only
returns strictly before that session. A refit that does not solve keeps the
previous target, so the month stays in the return series. The book starts
in cash, and it does not start until the first solved target.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data_handler import annualise_stats
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


def _as_weights(weights: np.ndarray | pd.Series, columns: pd.Index) -> np.ndarray:
    if isinstance(weights, pd.Series):
        aligned = weights.reindex(columns)
        if aligned.isna().any():
            missing = [str(col) for col in columns[aligned.isna()]]
            raise ValueError(f"weights are missing assets: {missing}")
        values = aligned.to_numpy(dtype=float)
    else:
        values = np.asarray(weights, dtype=float).ravel()
        if values.size != len(columns):
            raise ValueError(
                f"weights length {values.size} does not match {len(columns)} assets"
            )
    if not np.all(np.isfinite(values)) or np.any(values < -1e-12):
        raise ValueError("weights must be finite and long-only")
    values = np.clip(values, 0.0, None)
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("weights must sum to a positive number")
    return values / total


@dataclass(frozen=True)
class RebalanceResult:
    """Net simple returns of a book that drifts between scheduled trades.

    ``traded`` is the sum of absolute weight changes that day (dollars traded
    per dollar of NAV, buys and sells). ``turnover`` is half of that, the
    usual one-way number. ``costs`` is the fraction of NAV deducted before
    that day's asset return. From cash, the opening trade has ``traded`` of 1.
    """

    returns: pd.Series
    traded: pd.Series
    turnover: pd.Series
    costs: pd.Series
    weights: pd.DataFrame

    @property
    def total_traded(self) -> float:
        return float(self.traded.sum())

    @property
    def total_cost(self) -> float:
        """Sum of the daily cost fractions. This is not a compounded drag."""
        return float(self.costs.sum())


def run_scheduled_rebalance(
    returns: pd.DataFrame,
    targets: dict[pd.Timestamp, np.ndarray | pd.Series],
    cost_bps: float = 5.0,
    name: str | None = None,
) -> RebalanceResult:
    """Earn each day's simple return on the drifted weights, except on a target date.

    On a target date the book trades from the drifted weights (or from cash,
    before the first trade) to the target, pays ``cost_bps`` on each dollar
    bought or sold, then earns that day's return on the new weights.
    Leading sessions with no target yet are omitted: the strategy has not
    started, and those days are not reported as a zero return.
    """
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise ValueError("returns must be a non-empty DataFrame")
    if not np.isfinite(cost_bps) or cost_bps < 0.0:
        raise ValueError("cost_bps must be finite and non-negative")
    frame = returns.sort_index()
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("returns must be indexed by timestamp")
    if not np.all(np.isfinite(frame.to_numpy(dtype=float))):
        raise ValueError("returns must be finite")

    scheduled = {
        pd.Timestamp(stamp): _as_weights(target, frame.columns)
        for stamp, target in targets.items()
    }
    if not scheduled:
        raise ValueError("at least one target allocation is required")

    n_assets = frame.shape[1]
    current = np.zeros(n_assets)
    invested = False
    net_returns: list[float] = []
    traded_values: list[float] = []
    turnover_values: list[float] = []
    cost_values: list[float] = []
    weight_rows: list[np.ndarray] = []
    kept: list[pd.Timestamp] = []

    for stamp, row in frame.iterrows():
        stamp = pd.Timestamp(stamp)
        asset_returns = row.to_numpy(dtype=float)
        traded = 0.0
        if stamp in scheduled:
            target = scheduled[stamp]
            traded = float(np.abs(target - current).sum())
            current = target
            invested = True
        if not invested:
            continue
        cost_rate = traded * float(cost_bps) / 10_000.0
        gross = float(current @ asset_returns)
        if 1.0 + gross <= 0.0:
            raise ValueError("portfolio return is -100% or worse")
        net = (1.0 - cost_rate) * (1.0 + gross) - 1.0
        current = current * (1.0 + asset_returns) / (1.0 + gross)
        net_returns.append(net)
        traded_values.append(traded)
        turnover_values.append(0.5 * traded)
        cost_values.append(cost_rate)
        weight_rows.append(current.copy())
        kept.append(stamp)

    if not kept:
        raise ValueError("no session fell on or after a target allocation")
    index = pd.DatetimeIndex(kept)
    series = pd.Series(net_returns, index=index, name=name or "portfolio")
    return RebalanceResult(
        returns=series,
        traded=pd.Series(traded_values, index=index, name="traded"),
        turnover=pd.Series(turnover_values, index=index, name="turnover"),
        costs=pd.Series(cost_values, index=index, name="cost"),
        weights=pd.DataFrame(weight_rows, index=index, columns=frame.columns),
    )


def backtest_fixed_weights(
    returns: pd.DataFrame,
    weights: np.ndarray | pd.Series,
    cost_bps: float = 5.0,
    name: str | None = None,
) -> RebalanceResult:
    """Rebalance ``weights`` on the first session of each month.

    Between those sessions the holdings drift with prices. ``cost_bps`` is
    charged on dollars traded, including the opening trade from cash.
    """
    if returns.empty:
        raise ValueError("returns must be a non-empty DataFrame")
    target = _as_weights(weights, returns.columns)
    schedule = {start: target.copy() for start in _month_starts(returns.index)}
    return run_scheduled_rebalance(returns, schedule, cost_bps=cost_bps, name=name)


def backtest_sixty_forty(
    equity: pd.Series,
    bond: pd.Series,
    cost_bps: float = 5.0,
) -> RebalanceResult:
    """60% equity / 40% bond, monthly, on the dates both series have a return.

    This is the stock/bond 60/40. It is not 60% equity and 40% cash: cash at
    the same risk-free rate used in the Sharpe ratio has the equity Sharpe.
    """
    paired = pd.concat({"equity": equity, "bond": bond}, axis=1).dropna()
    if paired.empty:
        raise ValueError("equity and bond returns have no dates in common")
    return backtest_fixed_weights(
        paired,
        np.array([0.6, 0.4]),
        cost_bps=cost_bps,
        name="60/40",
    )


def walk_forward_max_sharpe(
    returns: pd.DataFrame,
    oos_start,
    risk_free_rate: float = 0.04,
    max_weight: float | None = None,
    min_history: int = 60,
    cost_bps: float = 5.0,
    shrink: bool = False,
) -> pd.Series:
    """Monthly refit of the long-only maximum-Sharpe portfolio, net of costs.

    For a rebalance session t, the fit uses ``returns`` with index < t.
    Weights then drift until the next month. A later month that cannot be
    solved is rebalanced to the previous target. Months before the first
    solved target are omitted, because there is no book yet. ``shrink``
    applies Ledoit-Wolf to the covariance only.
    """
    result = walk_forward_max_sharpe_result(
        returns,
        oos_start,
        risk_free_rate=risk_free_rate,
        max_weight=max_weight,
        min_history=min_history,
        cost_bps=cost_bps,
        shrink=shrink,
    )
    return result.returns


def walk_forward_max_sharpe_result(
    returns: pd.DataFrame,
    oos_start,
    risk_free_rate: float = 0.04,
    max_weight: float | None = None,
    min_history: int = 60,
    cost_bps: float = 5.0,
    shrink: bool = False,
) -> RebalanceResult:
    """Same backtest as ``walk_forward_max_sharpe``, with turnover and costs."""
    empty = RebalanceResult(
        returns=pd.Series(dtype=float, name="MSR (monthly rebalance)"),
        traded=pd.Series(dtype=float),
        turnover=pd.Series(dtype=float),
        costs=pd.Series(dtype=float),
        weights=pd.DataFrame(),
    )
    if returns.empty:
        return empty
    frame = returns.sort_index()
    oos = frame.loc[frame.index > pd.Timestamp(oos_start)]
    if oos.empty:
        return empty

    targets: dict[pd.Timestamp, np.ndarray] = {}
    last: pd.Series | None = None
    for start in _month_starts(oos.index):
        history = frame.loc[frame.index < start]
        if len(history) < min_history:
            continue
        mu, cov = annualise_stats(history, shrink=shrink)
        fitted = max_sharpe(mu, cov, risk_free_rate=risk_free_rate, max_weight=max_weight)
        if fitted["weights"] is None:
            if last is None:
                continue
            chosen = last
        else:
            chosen = fitted["weights"].reindex(frame.columns)
            last = chosen
        targets[start] = chosen.to_numpy(dtype=float)
    if not targets:
        return empty
    first = min(targets)
    window = oos.loc[oos.index >= first]
    result = run_scheduled_rebalance(
        window,
        targets,
        cost_bps=cost_bps,
        name="MSR (monthly rebalance)",
    )
    return result
