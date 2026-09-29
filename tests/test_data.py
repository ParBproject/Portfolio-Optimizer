"""Estimation window, simple-return moments, and price cleaning."""

import numpy as np
import pandas as pd
import pytest

from data.fetch_data import clean_prices, extract_close
from src.data_handler import (
    TRADING_DAYS,
    annualise_stats,
    compute_log_returns,
    compute_returns,
    prepare_price_data,
)
from src.metrics import TRADING_DAYS as METRICS_DAYS


def _prices(n=80, seed=1):
    idx = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 0.01, size=(n, 2))
    levels = 100 * np.exp(np.cumsum(noise, axis=0))
    return pd.DataFrame(levels, index=idx, columns=["A", "B"])


def test_annualisation_factor_is_shared():
    assert TRADING_DAYS == METRICS_DAYS == 252


def test_moments_use_simple_returns_and_sample_covariance():
    prices = _prices()
    simple = compute_returns(prices)
    mu, cov = annualise_stats(simple)
    assert mu.to_numpy() == pytest.approx((simple.mean() * 252).to_numpy())
    assert cov.to_numpy() == pytest.approx((simple.cov() * 252).to_numpy())
    log_mu = compute_log_returns(prices).mean() * 252
    assert not np.allclose(mu.to_numpy(), log_mu.to_numpy(), atol=1e-4)
    w = np.array([0.3, 0.7])
    realized = simple.to_numpy() @ w
    assert np.sqrt(w @ cov.to_numpy() @ w) == pytest.approx(
        float(np.std(realized, ddof=1) * np.sqrt(252))
    )


def test_split_on_a_closed_market_does_not_drop_the_next_session():
    idx = pd.bdate_range("2023-12-20", "2024-01-10")
    prices = pd.DataFrame(
        {"A": np.linspace(100, 110, len(idx)), "B": np.linspace(50, 55, len(idx))},
        index=idx,
    )
    # 2023-12-31 was a Sunday and is not in the business-day index.
    data = prepare_price_data(prices, train_end="2023-12-31")
    assert data["train_returns"].index.max() < data["test_returns"].index.min()
    first_oos = idx[idx > pd.Timestamp("2023-12-31")][0]
    assert data["test_returns"].index[0] == first_oos
    assert pd.Timestamp("2023-12-31") not in data["train_returns"].index


def test_estimates_ignore_prices_after_the_split():
    prices = _prices()
    split = prices.index[50]
    base = prepare_price_data(prices, train_end=split)
    bumped = prices.copy()
    bumped.iloc[51:] *= 3
    alt = prepare_price_data(bumped, train_end=split)
    pd.testing.assert_series_equal(base["mu"], alt["mu"])
    pd.testing.assert_frame_equal(base["cov"], alt["cov"])
    assert not np.allclose(base["test_returns"], alt["test_returns"])
    assert base["train_returns"].index.max() == split
    assert base["test_returns"].index.min() > split


def test_short_forward_fill_does_not_invent_a_long_halt():
    idx = pd.bdate_range("2020-01-01", periods=15)
    prices = pd.DataFrame(
        {
            "A": np.arange(15, dtype=float) + 10.0,
            "B": np.arange(15, dtype=float) + 30.0,
        },
        index=idx,
    )
    prices.iloc[3:5, 0] = np.nan
    prices.iloc[8:15, 1] = np.nan
    cleaned = clean_prices(prices, max_ffill=5)
    assert cleaned.loc[idx[3], "A"] == pytest.approx(12.0)
    assert cleaned.loc[idx[4], "A"] == pytest.approx(12.0)
    assert idx[12] in cleaned.index
    assert idx[13] not in cleaned.index
    assert cleaned.loc[idx[12], "B"] == pytest.approx(prices.iloc[7]["B"])


def test_leading_gap_is_not_backfilled():
    idx = pd.bdate_range("2020-01-01", periods=6)
    prices = pd.DataFrame({"A": np.arange(6, dtype=float) + 1.0, "B": np.arange(6, dtype=float) + 2.0}, index=idx)
    prices.iloc[0, 0] = np.nan
    cleaned = clean_prices(prices, max_ffill=5)
    assert idx[0] not in cleaned.index
    assert cleaned.iloc[0]["A"] == pytest.approx(2.0)


def test_extract_close_accepts_both_multiindex_layouts():
    idx = pd.bdate_range("2020-01-01", periods=3)
    field_first = pd.MultiIndex.from_product([["Close", "Open"], ["A", "B"]])
    raw = pd.DataFrame(np.arange(12, dtype=float).reshape(3, 4), index=idx, columns=field_first)
    prices = extract_close(raw, ["B", "A"])
    assert list(prices.columns) == ["B", "A"]

    ticker_first = pd.MultiIndex.from_product([["A", "B"], ["Open", "Close"]])
    raw = pd.DataFrame(np.arange(12, dtype=float).reshape(3, 4), index=idx, columns=ticker_first)
    prices = extract_close(raw, ["A", "B"])
    assert list(prices.columns) == ["A", "B"]
    flat = pd.DataFrame({"Close": [1.0, 2.0, 3.0]}, index=idx)
    assert list(extract_close(flat, ["AAPL"]).columns) == ["AAPL"]


def test_extract_close_missing_ticker_raises():
    idx = pd.bdate_range("2020-01-01", periods=3)
    cols = pd.MultiIndex.from_product([["Close"], ["A"]])
    raw = pd.DataFrame([[1.0], [2.0], [3.0]], index=idx, columns=cols)
    with pytest.raises(ValueError, match="MISSING"):
        extract_close(raw, ["A", "MISSING"])
