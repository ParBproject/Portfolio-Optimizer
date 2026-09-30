"""The monthly refit may use an expanding window, never the future."""

import numpy as np
import pandas as pd

from src.backtest import walk_forward_max_sharpe
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


def test_compile_app():
    import py_compile

    py_compile.compile("app.py", doraise=True)
