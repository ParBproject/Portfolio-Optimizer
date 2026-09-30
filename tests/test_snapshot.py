"""Bundled prices for the browser demo, and the live Yahoo path staying put."""

import pandas as pd
import pytest

from data.fetch_data import (
    BENCHMARK_TICKERS,
    DEFAULT_TICKERS,
    download_prices,
    exclusive_end,
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


def test_app_end_date_includes_the_last_snapshot_session(monkeypatch):
    monkeypatch.setenv("PORTFOLIO_PRICE_SOURCE", "snapshot")
    meta = snapshot_metadata()
    prices = download_prices(["SPY"], start="2024-12-01", end=exclusive_end(meta["last_session"]))
    assert exclusive_end("2024-12-31") == "2025-01-01"
    assert prices.index.max() == pd.Timestamp(meta["last_session"])


def test_snapshot_carries_the_bond_benchmark_on_the_same_sessions():
    meta = snapshot_metadata()
    assert BENCHMARK_TICKERS == ["AGG"]
    assert "AGG" in meta["tickers"]
    prices = pd.read_csv("data/snapshot/prices.csv", index_col=0, parse_dates=True)
    assert list(prices.columns) == meta["tickers"]
    assert prices["AGG"].notna().all()
    assert prices.index.min() == pd.Timestamp(meta["first_session"])
    assert prices.index.max() == pd.Timestamp(meta["last_session"])


def test_snapshot_gmvp_is_the_market_etf(monkeypatch):
    """No weight cap: long-only minimum variance on this universe is SPY."""
    monkeypatch.setenv("PORTFOLIO_PRICE_SOURCE", "snapshot")
    data = load_data(list(DEFAULT_TICKERS), "2015-01-01", "2025-01-01", train_end="2023-12-31")
    gmvp = min_variance(data["mu"], data["cov"], risk_free_rate=0.04)
    msr = max_sharpe(data["mu"], data["cov"], risk_free_rate=0.04)
    assert gmvp["weights"]["SPY"] == pytest.approx(1.0)
    assert gmvp["weights"].drop(labels=["SPY"]).max() == pytest.approx(0.0)
    # Pins the maximum-Sharpe mix so a change in annualisation or the program fails here.
    assert msr["weights"].to_numpy() == pytest.approx(
        [0.2339988, 0.4221424, 0.0, 0.2778131, 0.0660457, 0.0],
        abs=1e-6,
    )
    assert msr["sharpe"] == pytest.approx(0.9596655, abs=1e-6)


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
