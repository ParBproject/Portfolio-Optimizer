"""Ledoit-Wolf intensity, and the sample covariance staying the default."""

import numpy as np
import pandas as pd
import pytest

from src.covariance import ledoit_wolf
from src.data_handler import annualise_stats


def test_ledoit_wolf_matches_the_published_sklearn_example():
    # sklearn.covariance.ledoit_wolf_shrinkage docstring: this draw shrinks by 0.23.
    real_cov = np.array([[0.4, 0.2], [0.2, 0.8]])
    draws = np.random.RandomState(0).multivariate_normal(mean=[0.0, 0.0], cov=real_cov, size=50)
    frame = pd.DataFrame(draws, columns=["A", "B"])
    cov, shrinkage = ledoit_wolf(frame)
    assert shrinkage == pytest.approx(0.23, abs=0.005)
    centered = draws - draws.mean(axis=0)
    sample = (centered.T @ centered) / len(draws)
    mu = float(np.trace(sample) / 2)
    expected = (1.0 - shrinkage) * sample + shrinkage * mu * np.eye(2)
    assert cov.to_numpy() == pytest.approx(expected)
    assert np.linalg.eigvalsh(cov.to_numpy()).min() >= -1e-12


def test_sample_covariance_is_unchanged_unless_shrinkage_is_requested():
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(rng.normal(size=(40, 3)), columns=list("ABC"))
    mu, cov = annualise_stats(frame)
    mu_off, cov_off = annualise_stats(frame, shrink=False)
    _mu_on, cov_on = annualise_stats(frame, shrink=True)
    pd.testing.assert_series_equal(mu, mu_off)
    pd.testing.assert_frame_equal(cov, cov_off)
    assert not np.allclose(cov.to_numpy(), cov_on.to_numpy())
    shrunk, _delta = ledoit_wolf(frame)
    assert cov_on.to_numpy() == pytest.approx(shrunk.to_numpy() * 252)


def test_a_short_window_shrinks_harder_than_the_long_sample(monkeypatch):
    monkeypatch.setenv("PORTFOLIO_PRICE_SOURCE", "snapshot")
    from src.data_handler import load_data

    data = load_data(
        ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "SPY"],
        "2015-01-01",
        "2025-01-01",
        train_end="2023-12-31",
    )
    _long_cov, long_delta = ledoit_wolf(data["train_returns"])
    _short_cov, short_delta = ledoit_wolf(data["train_returns"].iloc[:60])
    # About 1% on nine years of six names. Well under a level that would move the frontier.
    assert 0.0 < long_delta < 0.02
    assert short_delta > long_delta
