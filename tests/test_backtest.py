"""The monthly refit may use an expanding window, never the future."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    backtest_fixed_weights,
    backtest_sixty_forty,
    walk_forward_max_sharpe,
)
from src.data_handler import annualise_stats
from src.optimizer import max_sharpe


def test_each_refit_uses_only_returns_before_that_session(monkeypatch):
    idx = pd.bdate_range("2020-01-02", "2020-06-30")
    rng = np.random.default_rng(0)
    returns = pd.DataFrame(
        {
            "A": rng.normal(0.001, 0.01, len(idx)),
            "B": rng.normal(-0.0002, 0.015, len(idx)),
        },
        index=idx,
    )
    oos_start = pd.Timestamp("2020-03-31")
    seen = []
    real = max_sharpe

    def spy(mu, cov, risk_free_rate=0.04, max_weight=None):
        seen.append(mu.copy())
        return real(mu, cov, risk_free_rate=risk_free_rate, max_weight=max_weight)

    monkeypatch.setattr("src.backtest.max_sharpe", spy)
    out = walk_forward_max_sharpe(returns, oos_start, risk_free_rate=0.0, min_history=40)
    assert not out.empty
    assert not out.index.duplicated().any()

    oos = returns.loc[returns.index > oos_start]
    starts = []
    seen_months = set()
    for ts, period in zip(oos.index, oos.index.to_period("M"), strict=True):
        if period not in seen_months:
            seen_months.add(period)
            starts.append(pd.Timestamp(ts))

    assert len(seen) == len(starts)
    for fitted_mu, start in zip(seen, starts, strict=True):
        history = returns.loc[returns.index < start]
        expected_mu, _expected_cov = annualise_stats(history)
        pd.testing.assert_series_equal(fitted_mu, expected_mu)
        assert out.loc[out.index >= start].index.min() >= start
        assert history.index.max() < start


def test_weights_drift_between_monthly_rebalances():
    # A daily constant-mix would earn 0.05 on the second day. Drifting earns 0.055.
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    returns = pd.DataFrame({"A": [0.10, 0.10], "B": [-0.10, 0.00]}, index=index)
    result = backtest_fixed_weights(returns, [0.5, 0.5], cost_bps=0.0)
    assert result.returns.iloc[0] == pytest.approx(0.0)
    assert result.returns.iloc[1] == pytest.approx(0.055)
    assert result.weights.iloc[0].to_numpy() == pytest.approx([0.55, 0.45])


def test_cost_is_charged_on_every_dollar_traded():
    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-02-01"])
    returns = pd.DataFrame({"A": [0.10, 0.0, 0.0], "B": [-0.10, 0.0, 0.0]}, index=index)
    result = backtest_fixed_weights(returns, [0.5, 0.5], cost_bps=10.0)
    # Open from cash: trade 100% of NAV, 10 bps, gross return 0.
    assert result.traded.iloc[0] == pytest.approx(1.0)
    assert result.returns.iloc[0] == pytest.approx(-0.001)
    assert result.traded.iloc[1] == pytest.approx(0.0)
    assert result.returns.iloc[1] == pytest.approx(0.0)
    # 0.55/0.45 drifted back to 0.50/0.50 trades 0.10 of NAV.
    assert result.traded.iloc[2] == pytest.approx(0.10)
    assert result.costs.iloc[2] == pytest.approx(0.0001)
    assert result.returns.iloc[2] == pytest.approx(-0.0001)
    assert result.turnover.iloc[2] == pytest.approx(0.05)


def test_sixty_forty_is_sixty_percent_equity_on_the_first_session():
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    equity = pd.Series([0.10, 0.00], index=index, name="SPY")
    bond = pd.Series([0.00, 0.02], index=index, name="AGG")
    result = backtest_sixty_forty(equity, bond, cost_bps=0.0)
    assert result.returns.iloc[0] == pytest.approx(0.06)
    # Equity grew, the bond did not, and nothing traded on the second session.
    assert result.returns.iloc[1] == pytest.approx(0.4 / 1.06 * 0.02)
    assert result.traded.iloc[1] == pytest.approx(0.0)


def test_a_failed_refit_keeps_the_previous_target(monkeypatch):
    index = pd.bdate_range("2020-01-02", "2020-04-30")
    returns = pd.DataFrame(0.0, index=index, columns=["A", "B"])
    march = index[index >= "2020-03-01"][0]
    returns.loc[march, "A"] = 0.10
    calls = {"n": 0}

    def fake(mu, cov, risk_free_rate=0.04, max_weight=None):
        calls["n"] += 1
        if calls["n"] == 2:
            return {
                "weights": None,
                "ret": np.nan,
                "vol": np.nan,
                "sharpe": np.nan,
                "status": "fail",
            }
        weight = 0.8 if calls["n"] == 1 else 0.2
        return {
            "weights": pd.Series({"A": weight, "B": 1.0 - weight}),
            "ret": 0.0,
            "vol": 0.1,
            "sharpe": 0.0,
            "status": "optimal",
        }

    monkeypatch.setattr("src.backtest.max_sharpe", fake)
    out = walk_forward_max_sharpe(
        returns,
        "2020-01-31",
        risk_free_rate=0.0,
        min_history=5,
        cost_bps=0.0,
    )
    oos = returns.loc[returns.index > pd.Timestamp("2020-01-31")]
    assert len(out) == len(oos)
    assert march in out.index
    # February's 80/20 is still on at the March open. April's 20/80 is not used early.
    # Returns are zero through February, so nothing has drifted.
    assert out.loc[march] == pytest.approx(0.08)
    assert calls["n"] == 3


def test_negative_cost_is_rejected():
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    returns = pd.DataFrame({"A": [0.01, 0.01]}, index=index)
    with pytest.raises(ValueError, match="cost_bps"):
        backtest_fixed_weights(returns, [1.0], cost_bps=-1.0)


def test_compile_app():
    import py_compile

    py_compile.compile("app.py", doraise=True)
