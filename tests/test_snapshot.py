"""Bundled prices for the browser demo, and the live Yahoo path staying put."""

import pandas as pd
import pytest

from data.fetch_data import (
    DEFAULT_TICKERS,
    download_prices,
    price_source,
    snapshot_label,
    snapshot_metadata,
)
from src.data_handler import load_data
from src.optimizer import efficient_frontier, max_sharpe, min_variance


def test_price_source_defaults_to_yahoo_outside_the_browser(monkeypatch):
    monkeypatch.delenv("PORTFOLIO_PRICE_SOURCE", raising=False)
    assert price_source() == "yahoo"


def test_snapshot_label_names_the_range():
    meta = snapshot_metadata()
    label = snapshot_label()
    assert "Static price snapshot" in label
    assert "not live Yahoo Finance" in label
    assert meta["first_session"] in label
    assert meta["last_session"] in label
    assert meta["downloaded_on"] in label
    for ticker in DEFAULT_TICKERS:
        assert ticker in label


def test_snapshot_download_does_not_call_yahoo(monkeypatch):
    monkeypatch.setenv("PORTFOLIO_PRICE_SOURCE", "snapshot")

    def _boom(*_args, **_kwargs):
        raise AssertionError("Yahoo Finance was called")

    monkeypatch.setattr("data.fetch_data._download_yahoo", _boom)
    prices = download_prices(["MSFT", "AAPL"], start="2018-01-01", end="2019-01-01")
    assert list(prices.columns) == ["MSFT", "AAPL"]
    assert prices.index.min() >= pd.Timestamp("2018-01-01")
    assert prices.index.max() < pd.Timestamp("2019-01-01")
    assert prices.index.is_monotonic_increasing


def test_snapshot_end_is_exclusive(monkeypatch):
    monkeypatch.setenv("PORTFOLIO_PRICE_SOURCE", "snapshot")
    meta = snapshot_metadata()
    last = meta["last_session"]
    inclusive = download_prices(["SPY"], start=meta["first_session"], end="2025-01-01")
    exclusive = download_prices(["SPY"], start=meta["first_session"], end=last)
    assert inclusive.index.max() == pd.Timestamp(last)
    assert exclusive.index.max() < pd.Timestamp(last)


def test_snapshot_rejects_unknown_tickers_and_empty_windows(monkeypatch):
    monkeypatch.setenv("PORTFOLIO_PRICE_SOURCE", "snapshot")
    with pytest.raises(ValueError, match="NFLX"):
        download_prices(["AAPL", "NFLX"], start="2018-01-01", end="2019-01-01")
    with pytest.raises(ValueError, match="snapshot"):
        download_prices(["AAPL"], start="1990-01-01", end="1991-01-01")


def test_snapshot_portfolios_match_the_cvxpy_solver(monkeypatch):
    monkeypatch.setenv("PORTFOLIO_PRICE_SOURCE", "snapshot")
    import src.optimizer as opt

    data = load_data(list(DEFAULT_TICKERS), "2015-01-01", "2025-01-01", train_end="2023-12-31")
    mu, cov = data["mu"], data["cov"]
    cvx_min = min_variance(mu, cov, risk_free_rate=0.04)
    cvx_sharpe = max_sharpe(mu, cov, risk_free_rate=0.04)
    frontier = efficient_frontier(mu, cov, n_points=6, risk_free_rate=0.04)
    assert cvx_min["weights"] is not None
    assert cvx_sharpe["weights"] is not None
    assert cvx_sharpe["sharpe"] + 1e-6 >= cvx_min["sharpe"]

    monkeypatch.setattr(opt, "cp", None)
    sci_min = opt.min_variance(mu, cov, risk_free_rate=0.04)
    sci_sharpe = opt.max_sharpe(mu, cov, risk_free_rate=0.04)
    sci_frontier = opt.efficient_frontier(mu, cov, n_points=6, risk_free_rate=0.04)
    assert sci_min["weights"].to_numpy() == pytest.approx(cvx_min["weights"].to_numpy(), abs=1e-5)
    assert sci_sharpe["weights"].to_numpy() == pytest.approx(
        cvx_sharpe["weights"].to_numpy(), abs=1e-5
    )
    assert sci_frontier[["ret", "vol"]].to_numpy() == pytest.approx(
        frontier[["ret", "vol"]].to_numpy(), abs=1e-5
    )
