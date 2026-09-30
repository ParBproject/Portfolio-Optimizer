"""Re-render the README charts from adjusted closes.

Uses the same defaults as the Streamlit app: AAPL, MSFT, GOOGL, AMZN, JPM,
and SPY, 2015-01-01 through 2024-12-31, training window through 2023-12-31,
risk-free rate 4%, no weight cap. Requires network access unless
data/cache already has the CSV. Kaleido writes the PNGs.

    python scripts/render_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data_handler import load_data, simulate_random_portfolios
from src.metrics import cumulative_wealth, portfolio_daily_returns
from src.optimizer import efficient_frontier, max_sharpe, min_variance
from src.visualization import (
    plot_backtest,
    plot_correlation_heatmap,
    plot_efficient_frontier,
    plot_weights,
)

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "SPY"]
START = "2015-01-01"
END = "2024-12-31"
TRAIN_END = "2023-12-31"
RISK_FREE = 0.04
OUT = Path(__file__).resolve().parents[1] / "screenshots"


def main() -> None:
    data = load_data(TICKERS, START, END, train_end=TRAIN_END)
    mu, cov = data["mu"], data["cov"]
    gmvp = min_variance(mu, cov, risk_free_rate=RISK_FREE)
    msr = max_sharpe(mu, cov, risk_free_rate=RISK_FREE)
    frontier = efficient_frontier(mu, cov, n_points=50, risk_free_rate=RISK_FREE)
    cloud = simulate_random_portfolios(mu, cov, n_portfolios=5000, risk_free_rate=RISK_FREE)
    if gmvp["weights"] is None or msr["weights"] is None or frontier.empty:
        raise RuntimeError("Optimiser failed; figures were not written.")

    OUT.mkdir(exist_ok=True)
    plot_efficient_frontier(
        frontier,
        cloud,
        gmvp,
        msr,
        data["tickers"],
        RISK_FREE,
        title="Markowitz Efficient Frontier",
    ).write_image(OUT / "01_efficient_frontier.png", scale=2)
    plot_weights(msr["weights"], "Maximum Sharpe — Weights").write_image(
        OUT / "02_portfolio_weights.png", scale=2
    )

    test = data["test_returns"]
    equal = pd.Series(1.0 / len(test.columns), index=test.columns)
    portfolios = {
        "GMVP": portfolio_daily_returns(gmvp["weights"], test),
        "Max Sharpe": portfolio_daily_returns(msr["weights"], test),
        "Equal Weight": portfolio_daily_returns(equal, test),
    }
    plot_backtest(
        {name: cumulative_wealth(series) for name, series in portfolios.items()}
    ).write_image(OUT / "03_backtest.png", scale=2)
    plot_correlation_heatmap(
        data["train_returns"],
        title="In-sample correlation (simple returns)",
    ).write_image(OUT / "04_correlations_stats.png", scale=2)
    print(f"Wrote figures to {OUT}")
    print(
        "Train",
        data["train_returns"].index.min().date(),
        "→",
        data["train_returns"].index.max().date(),
        "| test",
        data["test_returns"].index.min().date(),
        "→",
        data["test_returns"].index.max().date(),
    )


if __name__ == "__main__":
    main()
