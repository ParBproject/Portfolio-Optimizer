"""Re-render the README charts from adjusted closes.

Uses the same defaults as the Streamlit app: AAPL, MSFT, GOOGL, AMZN, JPM,
and SPY, 2015-01-01 through 2024-12-31 inclusive, training window through
2023-12-31, risk-free rate 4%, no weight cap. The backtest figure rebalances
monthly and charges 5 bps per dollar traded. Requires network access unless
``PORTFOLIO_PRICE_SOURCE=snapshot`` or ``data/cache`` already has the CSV.
Kaleido writes the PNGs.

    python scripts/render_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from data.fetch_data import BENCHMARK_TICKERS, download_prices, exclusive_end
from src.backtest import backtest_fixed_weights, backtest_sixty_forty
from src.data_handler import compute_returns, load_data, simulate_random_portfolios
from src.metrics import cumulative_wealth
from src.optimizer import efficient_frontier, max_sharpe, min_variance
from src.visualization import (
    plot_backtest,
    plot_correlation_heatmap,
    plot_efficient_frontier,
    plot_weights,
)

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "SPY"]
START = "2015-01-01"
END = "2024-12-31"  # last session included; download_prices wants the next day
TRAIN_END = "2023-12-31"
RISK_FREE = 0.04
COST_BPS = 5.0
OUT = Path(__file__).resolve().parents[1] / "screenshots"


def main() -> None:
    data = load_data(TICKERS, START, exclusive_end(END), train_end=TRAIN_END)
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
    books = {
        "GMVP": backtest_fixed_weights(test, gmvp["weights"], cost_bps=COST_BPS, name="GMVP"),
        "Max Sharpe": backtest_fixed_weights(
            test, msr["weights"], cost_bps=COST_BPS, name="Max Sharpe"
        ),
        "Equal Weight": backtest_fixed_weights(test, equal, cost_bps=COST_BPS, name="Equal Weight"),
        "SPY": backtest_fixed_weights(test[["SPY"]], [1.0], cost_bps=COST_BPS, name="SPY"),
    }
    bond_prices = download_prices(list(BENCHMARK_TICKERS), start=START, end=exclusive_end(END))
    bond = compute_returns(bond_prices)[BENCHMARK_TICKERS[0]]
    books["60/40"] = backtest_sixty_forty(test["SPY"], bond, cost_bps=COST_BPS)
    plot_backtest(
        {name: cumulative_wealth(book.returns) for name, book in books.items()},
        title="Out-of-sample wealth (monthly rebalance, net of costs)",
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
