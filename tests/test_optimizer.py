"""Closed-form portfolios, constraints, risk-free rate, and failed solves."""

import numpy as np
import pandas as pd
import pytest

from src.metrics import annualised_return, annualised_volatility, sharpe_ratio
from src.optimizer import (
    efficient_frontier,
    max_sharpe,
    max_sharpe_scipy,
    min_variance,
    min_variance_scipy,
)
from src.visualization import plot_efficient_frontier


def _two_asset():
    mu = pd.Series([0.10, 0.05], index=["A", "B"])
    cov = pd.DataFrame([[0.04, 0.0], [0.0, 0.01]], index=["A", "B"], columns=["A", "B"])
    return mu, cov


def test_gmvp_matches_the_diagonal_closed_form():
    mu, cov = _two_asset()
    result = min_variance(mu, cov, risk_free_rate=0.02)
    assert result["weights"] is not None
    assert result["weights"].sum() == pytest.approx(1.0)
    assert result["weights"]["A"] == pytest.approx(0.2, abs=1e-6)
    assert result["weights"]["B"] == pytest.approx(0.8, abs=1e-6)
    assert result["ret"] == pytest.approx(0.06, abs=1e-6)
    assert result["vol"] == pytest.approx(np.sqrt(0.008), abs=1e-6)
    assert result["sharpe"] == pytest.approx((0.06 - 0.02) / np.sqrt(0.008), abs=1e-6)


def test_gmvp_sharpe_uses_the_supplied_risk_free_rate():
    mu, cov = _two_asset()
    low = min_variance(mu, cov, risk_free_rate=0.0)
    high = min_variance(mu, cov, risk_free_rate=0.03)
    assert low["weights"].to_numpy() == pytest.approx(high["weights"].to_numpy())
    assert low["sharpe"] - high["sharpe"] == pytest.approx(0.03 / low["vol"])


def test_max_sharpe_matches_the_closed_form():
    mu, cov = _two_asset()
    result = max_sharpe(mu, cov, risk_free_rate=0.0)
    assert result["weights"]["A"] == pytest.approx(1.0 / 3.0, abs=1e-5)
    assert result["weights"]["B"] == pytest.approx(2.0 / 3.0, abs=1e-5)
    assert result["weights"].sum() == pytest.approx(1.0)
    priced = max_sharpe(mu, cov, risk_free_rate=0.02)
    assert priced["weights"]["A"] == pytest.approx(0.4, abs=1e-5)
    assert priced["weights"]["B"] == pytest.approx(0.6, abs=1e-5)


def test_target_return_portfolio():
    mu, cov = _two_asset()
    result = min_variance(mu, cov, target_return=0.08, risk_free_rate=0.0)
    assert result["weights"]["A"] == pytest.approx(0.6, abs=1e-5)
    assert result["weights"]["B"] == pytest.approx(0.4, abs=1e-5)
    assert result["vol"] == pytest.approx(np.sqrt(0.016), abs=1e-5)


def test_scipy_matches_cvxpy_on_the_closed_form():
    mu, cov = _two_asset()
    cvx = min_variance(mu, cov, risk_free_rate=0.01)
    sci = min_variance_scipy(mu, cov, risk_free_rate=0.01)
    assert sci["weights"].to_numpy() == pytest.approx(cvx["weights"].to_numpy(), abs=1e-5)
    cvx_s = max_sharpe(mu, cov, risk_free_rate=0.0)
    sci_s = max_sharpe_scipy(mu, cov, risk_free_rate=0.0)
    assert sci_s["weights"].to_numpy() == pytest.approx(cvx_s["weights"].to_numpy(), abs=1e-4)


def test_weight_cap_is_enforced_and_sums_to_one():
    mu, cov = _two_asset()
    capped = max_sharpe(mu, cov, risk_free_rate=0.0, max_weight=0.5)
    assert capped["weights"].to_numpy() == pytest.approx([0.5, 0.5], abs=1e-5)
    assert (capped["weights"] <= 0.5 + 1e-8).all()


def test_infeasible_cap_and_target_do_not_raise():
    mu, cov = _two_asset()
    cap = min_variance(mu, cov, max_weight=0.1)
    assert cap["weights"] is None
    assert cap["status"] == "infeasible_max_weight"
    target = min_variance(mu, cov, target_return=5.0)
    assert target["weights"] is None
    assert target["status"] == "infeasible_target_return"
    underwater = max_sharpe(pd.Series([0.01, 0.02]), cov, risk_free_rate=0.05)
    assert underwater["weights"] is None
    assert underwater["status"] == "no_positive_excess_return"


def test_indefinite_covariance_is_rejected():
    mu = pd.Series([0.1, 0.1], index=["A", "B"])
    cov = pd.DataFrame([[1.0, 2.0], [2.0, 1.0]], index=["A", "B"], columns=["A", "B"])
    result = min_variance(mu, cov)
    assert result["weights"] is None
    assert result["status"] == "covariance_not_psd"


def test_frontier_respects_the_weight_cap():
    mu = pd.Series([0.05, 0.10, 0.20], index=["A", "B", "C"])
    cov = pd.DataFrame(
        np.diag([0.04, 0.04, 0.09]),
        index=mu.index,
        columns=mu.index,
    )
    frontier = efficient_frontier(mu, cov, n_points=12, max_weight=0.4, risk_free_rate=0.01)
    assert len(frontier) >= 8
    # 0.4*0.20 + 0.4*0.10 + 0.2*0.05 = 0.13, with a small interior buffer.
    assert frontier["ret"].max() <= 0.13 + 1e-6
    weights = frontier[["A", "B", "C"]]
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert (weights.to_numpy() <= 0.4 + 1e-4).all()
    assert (weights.to_numpy() >= -1e-8).all()


def test_max_sharpe_beats_random_long_only_portfolios():
    rng = np.random.default_rng(2)
    mu = pd.Series([0.08, 0.12, 0.05], index=["A", "B", "C"])
    corr = np.array([[1.0, 0.2, 0.1], [0.2, 1.0, 0.3], [0.1, 0.3, 1.0]])
    vol = np.array([0.15, 0.22, 0.10])
    cov = pd.DataFrame(np.outer(vol, vol) * corr, index=mu.index, columns=mu.index)
    best = max_sharpe(mu, cov, risk_free_rate=0.02)
    gmvp = min_variance(mu, cov, risk_free_rate=0.02)
    assert best["sharpe"] + 1e-6 >= gmvp["sharpe"]
    for _ in range(25):
        w = rng.dirichlet(np.ones(3))
        ret = float(w @ mu.to_numpy())
        sigma = float(np.sqrt(w @ cov.to_numpy() @ w))
        assert (ret - 0.02) / sigma <= best["sharpe"] + 1e-4


def test_optimised_sharpe_matches_the_realized_sample():
    rng = np.random.default_rng(4)
    draws = pd.DataFrame(rng.normal(0.0004, 0.01, size=(400, 3)), columns=list("ABC"))
    from src.data_handler import annualise_stats

    mu, cov = annualise_stats(draws)
    rf = 0.02
    result = min_variance(mu, cov, risk_free_rate=rf)
    from src.metrics import portfolio_daily_returns

    portfolio = portfolio_daily_returns(result["weights"], draws)
    assert annualised_return(portfolio) == pytest.approx(result["ret"], abs=1e-8)
    assert annualised_volatility(portfolio) == pytest.approx(result["vol"], abs=1e-8)
    assert sharpe_ratio(portfolio, rf) == pytest.approx(result["sharpe"], abs=1e-8)


def test_browser_solver_matches_cvxpy(monkeypatch):
    """Pyodide has no CVXPY. SLSQP has to hit the same portfolios."""
    import src.optimizer as opt

    mu, cov = _two_asset()
    cvx_min = opt.min_variance(mu, cov, risk_free_rate=0.02)
    cvx_sharpe = opt.max_sharpe(mu, cov, risk_free_rate=0.02)
    cvx_capped = opt.max_sharpe(mu, cov, risk_free_rate=0.0, max_weight=0.5)
    frontier = opt.efficient_frontier(mu, cov, n_points=8, risk_free_rate=0.02)

    monkeypatch.setattr(opt, "cp", None)
    sci_min = opt.min_variance(mu, cov, risk_free_rate=0.02)
    sci_sharpe = opt.max_sharpe(mu, cov, risk_free_rate=0.02)
    sci_capped = opt.max_sharpe(mu, cov, risk_free_rate=0.0, max_weight=0.5)
    sci_frontier = opt.efficient_frontier(mu, cov, n_points=8, risk_free_rate=0.02)
    blocked = opt.min_variance(mu, cov, max_weight=0.1)
    underwater = opt.max_sharpe(pd.Series([0.01, 0.02]), cov, risk_free_rate=0.05)

    assert sci_min["weights"].to_numpy() == pytest.approx(cvx_min["weights"].to_numpy(), abs=1e-5)
    assert sci_sharpe["weights"].to_numpy() == pytest.approx(
        cvx_sharpe["weights"].to_numpy(), abs=1e-4
    )
    assert sci_capped["weights"].to_numpy() == pytest.approx(cvx_capped["weights"].to_numpy(), abs=1e-4)
    assert sci_frontier[["ret", "vol"]].to_numpy() == pytest.approx(
        frontier[["ret", "vol"]].to_numpy(), abs=1e-4
    )
    assert blocked["weights"] is None
    assert blocked["status"] == "infeasible_max_weight"
    assert underwater["weights"] is None
    assert underwater["status"] == "no_positive_excess_return"


def test_dark_demo_charts_use_the_emerald_accent():
    mu, cov = _two_asset()
    gmvp = min_variance(mu, cov, risk_free_rate=0.01)
    msr = max_sharpe(mu, cov, risk_free_rate=0.01)
    frontier = efficient_frontier(mu, cov, n_points=8, risk_free_rate=0.01)
    fig = plot_efficient_frontier(frontier, None, gmvp, msr, risk_free_rate=0.01, theme="dark")
    curve = next(trace for trace in fig.data if trace.name == "Efficient Frontier")
    assert curve.line.color == "#10B981"
    assert fig.layout.paper_bgcolor == "#0B1220"
    assert fig.layout.plot_bgcolor == "#111827"


def test_capital_allocation_line_stops_at_the_tangency_portfolio():
    mu, cov = _two_asset()
    gmvp = min_variance(mu, cov, risk_free_rate=0.01)
    msr = max_sharpe(mu, cov, risk_free_rate=0.01)
    frontier = efficient_frontier(mu, cov, n_points=8, risk_free_rate=0.01)
    fig = plot_efficient_frontier(frontier, None, gmvp, msr, risk_free_rate=0.01)
    line = next(trace for trace in fig.data if trace.name == "Capital allocation line")
    assert max(line.x) == pytest.approx(msr["vol"])
