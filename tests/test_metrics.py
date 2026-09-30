"""Pins for return convention, Sharpe, drawdown, and weight alignment."""

import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    TRADING_DAYS,
    annualised_return,
    annualised_volatility,
    cagr,
    calmar_ratio,
    compare_portfolios,
    cumulative_wealth,
    format_comparison,
    max_drawdown,
    portfolio_daily_returns,
    sharpe_ratio,
)


def test_sharpe_uses_arithmetic_excess_return_not_cagr():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0008, 0.01, size=300)
    rf = 0.03
    vol = float(np.std(r, ddof=1) * np.sqrt(TRADING_DAYS))
    expected = (float(np.mean(r) * TRADING_DAYS) - rf) / vol
    assert annualised_volatility(r) == pytest.approx(vol)
    assert sharpe_ratio(r, rf) == pytest.approx(expected)
    cagr_sharpe = (cagr(r) - rf) / vol
    assert sharpe_ratio(r, rf) != pytest.approx(cagr_sharpe, rel=1e-3)


def test_volatility_uses_sample_not_population_std():
    r = np.array([0.01, -0.02, 0.015, 0.0, -0.005])
    population = float(np.std(r, ddof=0) * np.sqrt(TRADING_DAYS))
    sample = float(np.std(r, ddof=1) * np.sqrt(TRADING_DAYS))
    assert annualised_volatility(r) == pytest.approx(sample)
    assert annualised_volatility(r) != pytest.approx(population)


def test_zero_volatility_sharpe_is_nan():
    assert np.isnan(sharpe_ratio(np.zeros(20), 0.02))


def test_max_drawdown_counts_the_drop_from_starting_wealth():
    # Wealth path: 1.00 -> 0.90 -> 0.945. Ignoring the start reports 0.
    r = np.array([-0.10, 0.05])
    wealth = cumulative_wealth(r)
    assert wealth.iloc[0] == pytest.approx(1.0)
    assert wealth.iloc[-1] == pytest.approx(0.90 * 1.05)
    assert max_drawdown(wealth) == pytest.approx(0.10)


def test_calmar_is_cagr_over_drawdown_without_the_risk_free_rate():
    r = np.array([0.01, -0.02, 0.005, 0.004, -0.01, 0.008])
    drawdown = max_drawdown(cumulative_wealth(r))
    assert calmar_ratio(r) == pytest.approx(cagr(r) / drawdown)
    excess_version = (cagr(r) - 0.04) / drawdown
    assert calmar_ratio(r) != pytest.approx(excess_version)


def test_portfolio_return_is_the_weighted_sum_of_simple_returns():
    simple = pd.DataFrame({"A": [0.10, -0.05], "B": [0.02, 0.04]})
    weights = pd.Series({"B": 0.25, "A": 0.75})
    portfolio = portfolio_daily_returns(weights, simple)
    expected = 0.75 * simple["A"] + 0.25 * simple["B"]
    pd.testing.assert_series_equal(portfolio, expected, check_names=False)
    log_dot = np.log1p(simple).to_numpy() @ np.array([0.75, 0.25])
    assert not np.allclose(portfolio.to_numpy(), log_dot)


def test_weights_far_from_one_are_rejected():
    simple = pd.DataFrame({"A": [0.01], "B": [0.02]})
    with pytest.raises(ValueError, match="sum to 1"):
        portfolio_daily_returns(np.array([0.2, 0.2]), simple)


def test_comparison_table_formats_without_jinja():
    returns = {
        "A": pd.Series([0.01, -0.02, 0.015]),
        "B": pd.Series([0.0, 0.0, 0.0]),
    }
    table, formats = compare_portfolios(returns, risk_free_rate=0.01)
    display = format_comparison(table, formats)
    assert display.loc["Arithmetic Return", "A"].endswith("%")
    assert display.loc["Sharpe Ratio", "A"] == f"{table.loc['Sharpe Ratio', 'A']:.3f}"
    assert display.loc["Sharpe Ratio", "B"] == ""


def test_annualised_return_is_arithmetic():
    r = np.array([0.01, -0.02, 0.015])
    assert annualised_return(r) == pytest.approx(float(np.mean(r) * TRADING_DAYS))
    assert annualised_return(r) != pytest.approx(cagr(r))
