"""Historical and walk-forward portfolio backtesting.

The functions in this module keep estimation data strictly before each
rebalance date, making the rolling workflow suitable for demonstrating
out-of-sample quantitative research discipline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .optimizer import max_sharpe, min_variance

TRADING_DAYS = 252


@dataclass(frozen=True)
class BacktestResult:
    """Net portfolio history plus implementation diagnostics."""

    returns: pd.Series
    wealth: pd.Series
    turnover: pd.Series
    transaction_costs: pd.Series
    weights: pd.DataFrame

    @property
    def total_turnover(self) -> float:
        return float(self.turnover.sum())

    @property
    def total_transaction_cost(self) -> float:
        return float(self.transaction_costs.sum())


def _validate_weights(weights: np.ndarray | pd.Series, n_assets: int) -> np.ndarray:
    values = np.asarray(weights, dtype=float)
    if values.shape != (n_assets,):
        raise ValueError("weights must match the number of assets")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("weights must be finite and non-negative")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("weights must have a positive total")
    return values / total


def _simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty or prices.shape[1] == 0:
        raise ValueError("prices must contain at least one asset")
    numeric = prices.apply(pd.to_numeric, errors="coerce").ffill().dropna()
    if len(numeric) < 2:
        raise ValueError("prices must contain at least two complete observations")
    if (numeric <= 0).any(axis=None):
        raise ValueError("prices must be strictly positive")
    return numeric.pct_change(fill_method=None).dropna()


def _run_target_schedule(
    returns: pd.DataFrame,
    targets: dict[pd.Timestamp, np.ndarray],
    *,
    transaction_cost_bps: float,
    initial_value: float,
) -> BacktestResult:
    if transaction_cost_bps < 0 or not np.isfinite(transaction_cost_bps):
        raise ValueError("transaction_cost_bps must be finite and non-negative")
    if initial_value <= 0 or not np.isfinite(initial_value):
        raise ValueError("initial_value must be finite and positive")
    if not targets:
        raise ValueError("at least one target allocation is required")

    first_target = next(iter(targets.values()))
    current_weights = first_target.copy()
    wealth_value = float(initial_value)

    net_returns: list[float] = []
    wealth_values: list[float] = []
    turnover_values: list[float] = []
    cost_values: list[float] = []
    weight_rows: list[np.ndarray] = []

    for timestamp, row in returns.iterrows():
        turnover = 0.0
        transaction_cost = 0.0

        if timestamp in targets:
            target = targets[timestamp]
            turnover = 0.5 * float(np.abs(target - current_weights).sum())
            cost_rate = turnover * transaction_cost_bps / 10_000.0
            transaction_cost = wealth_value * cost_rate
            wealth_value -= transaction_cost
            current_weights = target.copy()

        starting_wealth = wealth_value
        asset_returns = row.to_numpy(dtype=float)
        gross_return = float(current_weights @ asset_returns)
        wealth_value *= 1.0 + gross_return
        if wealth_value < 0:
            raise ValueError("portfolio wealth became negative")

        denominator = 1.0 + gross_return
        if denominator <= 0:
            raise ValueError("portfolio experienced a return of -100% or worse")
        current_weights = current_weights * (1.0 + asset_returns) / denominator

        net_return = (
            wealth_value / (starting_wealth + transaction_cost) - 1.0
            if starting_wealth + transaction_cost > 0
            else 0.0
        )
        # Fold implementation cost into the period return.
        if transaction_cost > 0:
            prior_wealth = starting_wealth + transaction_cost
            net_return = wealth_value / prior_wealth - 1.0

        net_returns.append(net_return)
        wealth_values.append(wealth_value)
        turnover_values.append(turnover)
        cost_values.append(transaction_cost)
        weight_rows.append(current_weights.copy())

    index = returns.index
    return BacktestResult(
        returns=pd.Series(net_returns, index=index, name="portfolio_return"),
        wealth=pd.Series(wealth_values, index=index, name="portfolio_wealth"),
        turnover=pd.Series(turnover_values, index=index, name="turnover"),
        transaction_costs=pd.Series(cost_values, index=index, name="transaction_cost"),
        weights=pd.DataFrame(weight_rows, index=index, columns=returns.columns),
    )


def backtest_fixed_allocation(
    prices: pd.DataFrame,
    weights: np.ndarray | pd.Series,
    *,
    rebalance_every: int = 21,
    transaction_cost_bps: float = 5.0,
    initial_value: float = 1.0,
) -> BacktestResult:
    """Backtest a fixed target allocation with periodic rebalancing."""
    if isinstance(rebalance_every, bool) or not isinstance(rebalance_every, int):
        raise TypeError("rebalance_every must be a positive integer")
    if rebalance_every < 1:
        raise ValueError("rebalance_every must be a positive integer")

    returns = _simple_returns(prices)
    target = _validate_weights(weights, returns.shape[1])
    schedule = {
        timestamp: target.copy()
        for position, timestamp in enumerate(returns.index)
        if position % rebalance_every == 0
    }
    return _run_target_schedule(
        returns,
        schedule,
        transaction_cost_bps=transaction_cost_bps,
        initial_value=initial_value,
    )


def walk_forward_backtest(
    prices: pd.DataFrame,
    *,
    strategy: str = "max_sharpe",
    lookback_days: int = 252,
    rebalance_every: int = 21,
    risk_free_rate: float = 0.04,
    max_weight: float | None = None,
    transaction_cost_bps: float = 5.0,
    initial_value: float = 1.0,
) -> BacktestResult:
    """Run rolling no-lookahead portfolio optimization.

    At each rebalance date, expected returns and covariance are estimated only
    from the preceding lookback window. The optimized weights are then held
    until the next rebalance.
    """
    if strategy not in {"max_sharpe", "min_variance"}:
        raise ValueError("strategy must be 'max_sharpe' or 'min_variance'")
    if isinstance(lookback_days, bool) or not isinstance(lookback_days, int):
        raise TypeError("lookback_days must be a positive integer")
    if lookback_days < 20:
        raise ValueError("lookback_days must be at least 20")
    if isinstance(rebalance_every, bool) or not isinstance(rebalance_every, int):
        raise TypeError("rebalance_every must be a positive integer")
    if rebalance_every < 1:
        raise ValueError("rebalance_every must be a positive integer")

    returns = _simple_returns(prices)
    if len(returns) <= lookback_days:
        raise ValueError("price history is too short for the requested lookback")

    schedule: dict[pd.Timestamp, np.ndarray] = {}
    for position in range(lookback_days, len(returns), rebalance_every):
        estimation = returns.iloc[position - lookback_days : position]
        log_returns = np.log1p(estimation)
        mu = log_returns.mean() * TRADING_DAYS
        cov = log_returns.cov() * TRADING_DAYS

        if strategy == "max_sharpe":
            result = max_sharpe(
                mu,
                cov,
                risk_free_rate=risk_free_rate,
                max_weight=max_weight,
            )
        else:
            result = min_variance(mu, cov, max_weight=max_weight)

        weights = result.get("weights")
        if weights is None:
            continue
        schedule[returns.index[position]] = _validate_weights(
            weights,
            returns.shape[1],
        )

    if not schedule:
        raise ValueError("optimizer did not produce any feasible rebalance weights")

    first_rebalance = next(iter(schedule))
    live_returns = returns.loc[first_rebalance:]
    return _run_target_schedule(
        live_returns,
        schedule,
        transaction_cost_bps=transaction_cost_bps,
        initial_value=initial_value,
    )
