"""
app.py  –  Streamlit interactive dashboard
==========================================
Launch with:
    streamlit run app.py

Provides a browser-based UI for:
  • Configuring tickers, date range, and constraints
  • Computing and visualising the Efficient Frontier
  • Viewing GMVP and Max-Sharpe portfolios
  • Running a simple out-of-sample backtest
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd

from data.fetch_data import (
    BENCHMARK_TICKERS, download_prices, exclusive_end, price_source, snapshot_label,
)
from src.backtest import backtest_fixed_weights, backtest_sixty_forty
from src.covariance import ledoit_wolf
from src.data_handler import TRADING_DAYS, compute_returns, load_data, simulate_random_portfolios
from src.optimizer    import min_variance, max_sharpe, efficient_frontier
from src.metrics      import (
    portfolio_daily_returns, cumulative_wealth, compare_portfolios, format_comparison,
)
from src.visualization import (
    plot_efficient_frontier, plot_weights, plot_backtest,
    plot_drawdown, plot_correlation_heatmap,
)

# The browser demo cannot reach Yahoo Finance. Local runs leave this false
# and keep the live download path, including the light chart colours used
# for the README figures.
_SNAPSHOT = price_source() == "snapshot"
_THEME = "dark" if _SNAPSHOT else "light"


def _bond_returns(start: str, end_exclusive: str) -> pd.Series | None:
    """AGG simple returns for the 60/40 sleeve. Missing data omits the benchmark."""
    try:
        prices = download_prices(list(BENCHMARK_TICKERS), start=start, end=end_exclusive)
    except Exception:
        return None
    column = BENCHMARK_TICKERS[0]
    returns = compute_returns(prices)
    if column not in returns.columns or returns.empty:
        return None
    return returns[column]

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 Markowitz Portfolio Optimizer")
st.markdown(
    "Interactive mean-variance optimisation: Efficient Frontier, "
    "Global Minimum Variance, and Maximum Sharpe Ratio portfolios."
)

if _SNAPSHOT:
    st.warning(snapshot_label())

# ── Sidebar inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    if _SNAPSHOT:
        st.caption(
            "Browser demo: only the bundled snapshot is available. "
            "Its date range is shown above. Other symbols need a local run."
        )

    tickers_raw = st.text_input(
        "Tickers (comma-separated)",
        value="AAPL, MSFT, GOOGL, AMZN, JPM, SPY",
    )
    tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]

    col1, col2 = st.columns(2)
    start_date = col1.date_input("Start date", value=pd.Timestamp("2015-01-01"))
    end_date   = col2.date_input("End date (included)", value=pd.Timestamp("2024-12-31"))
    train_end  = st.date_input("Train/test split date", value=pd.Timestamp("2023-12-31"))

    st.subheader("Constraints")
    risk_free = st.slider("Risk-free rate (%)", 0.0, 8.0, 4.0, 0.25) / 100
    max_w = st.slider("Max weight per asset (0 = no cap)", 0.0, 1.0, 0.0, 0.05)
    max_weight = max_w if max_w > 0.0 else None
    covariance = st.selectbox(
        "Covariance",
        ["Sample", "Ledoit-Wolf"],
        help=(
            "Sample is the unbiased covariance (divide by T − 1). "
            "Ledoit-Wolf shrinks the maximum-likelihood covariance toward "
            "average variance. Expected returns stay the sample mean either way."
        ),
    )
    cost_bps = st.slider(
        "Trade cost (bps per dollar traded)",
        0.0, 25.0, 5.0, 0.5,
        help="Charged on buys and sells, including the opening trade from cash.",
    )
    n_frontier = st.slider("Frontier points", 20, 100, 50, 5)
    n_random   = st.slider("Random portfolios (cloud)", 1000, 10000, 5000, 500)

    run_btn = st.button("🚀 Optimise", type="primary", use_container_width=True)

# ── Main logic ─────────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner("Fetching data & optimising…"):
        try:
            data = load_data(
                tickers,
                start=str(start_date),
                end=exclusive_end(end_date),
                train_end=str(train_end),
            )
        except Exception as e:
            st.error(f"Data error: {e}")
            st.stop()

        mu  = data["mu"]
        cov = data["cov"]
        shrinkage = None
        if covariance == "Ledoit-Wolf":
            shrunk, shrinkage = ledoit_wolf(data["train_returns"])
            cov = shrunk * TRADING_DAYS
        tickers = data["tickers"]

        # ── Optimisation ──────────────────────────────────────────────────────
        try:
            gmvp = min_variance(mu, cov, max_weight=max_weight, risk_free_rate=risk_free)
            msr  = max_sharpe(mu, cov, risk_free_rate=risk_free, max_weight=max_weight)
            ef   = efficient_frontier(
                mu, cov, n_points=n_frontier, max_weight=max_weight, risk_free_rate=risk_free,
            )
            rand = simulate_random_portfolios(
                mu, cov, n_portfolios=n_random, risk_free_rate=risk_free,
            )
        except Exception as e:
            st.error(f"Optimisation error: {e}")
            st.stop()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🗺️ Efficient Frontier", "⚖️ Portfolios", "📊 Backtest", "🔥 Correlations"]
    )

    # ── Tab 1: Frontier ────────────────────────────────────────────────────────
    with tab1:
        fig = plot_efficient_frontier(ef, rand, gmvp, msr, tickers, risk_free, theme=_THEME)
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2: Portfolio details ───────────────────────────────────────────────
    with tab2:
        st.caption(
            "Return, volatility, and Sharpe in this tab are in-sample expected "
            "moments of the training window, not a track record. "
            "Out-of-sample wealth is on the Backtest tab."
            + (
                f" Ledoit-Wolf intensity toward average variance: {shrinkage:.1%}."
                if shrinkage is not None
                else ""
            )
        )
        col_g, col_s = st.columns(2)
        with col_g:
            st.subheader("🟢 Global Minimum Variance")
            if gmvp["weights"] is not None:
                st.metric("In-sample return", f"{gmvp['ret']:.2%}")
                st.metric("In-sample vol", f"{gmvp['vol']:.2%}")
                st.metric("In-sample Sharpe", f"{gmvp['sharpe']:.3f}")
                top = gmvp["weights"].idxmax()
                if gmvp["weights"].max() >= 1.0 - 1e-8:
                    st.caption(f"This portfolio is entirely {top}.")
                st.plotly_chart(
                    plot_weights(gmvp["weights"], "GMVP Weights", theme=_THEME),
                    use_container_width=True,
                )
            else:
                st.warning(f"Minimum-variance portfolio was not solved ({gmvp.get('status')}).")

        with col_s:
            st.subheader("🔴 Maximum Sharpe Ratio")
            if msr["weights"] is not None:
                st.metric("In-sample return", f"{msr['ret']:.2%}")
                st.metric("In-sample vol", f"{msr['vol']:.2%}")
                st.metric("In-sample Sharpe", f"{msr['sharpe']:.3f}")
                st.plotly_chart(
                    plot_weights(msr["weights"], "MSR Weights", theme=_THEME),
                    use_container_width=True,
                )
            else:
                st.warning(f"Maximum-Sharpe portfolio was not solved ({msr.get('status')}).")

    # ── Tab 3: Backtest ────────────────────────────────────────────────────────
    with tab3:
        test_returns = data["test_returns"]
        if test_returns.empty:
            st.warning("No out-of-sample data. Adjust the train/test split date.")
        else:
            books = {}
            if gmvp["weights"] is not None:
                books["GMVP"] = backtest_fixed_weights(
                    test_returns, gmvp["weights"], cost_bps=cost_bps, name="GMVP",
                )
            if msr["weights"] is not None:
                books["Max Sharpe"] = backtest_fixed_weights(
                    test_returns, msr["weights"], cost_bps=cost_bps, name="Max Sharpe",
                )
            equal = pd.Series(1.0 / len(tickers), index=test_returns.columns)
            books["Equal Weight"] = backtest_fixed_weights(
                test_returns, equal, cost_bps=cost_bps, name="Equal Weight",
            )
            if "SPY" in test_returns.columns:
                books["SPY"] = backtest_fixed_weights(
                    test_returns[["SPY"]], [1.0], cost_bps=cost_bps, name="SPY",
                )
                bond = _bond_returns(str(start_date), exclusive_end(end_date))
                if bond is None:
                    st.warning("60/40 skipped: AGG prices were not available for this window.")
                else:
                    try:
                        books["60/40"] = backtest_sixty_forty(
                            test_returns["SPY"], bond, cost_bps=cost_bps,
                        )
                    except ValueError as exc:
                        st.warning(f"60/40 benchmark skipped: {exc}")

            portfolios_dr = {name: book.returns for name, book in books.items()}
            # The optimiser's expected return is the gross weighted sum, which is
            # a daily rebalance with no costs. Keep it visible and named as such.
            if msr["weights"] is not None:
                portfolios_dr["Max Sharpe (daily, no costs)"] = portfolio_daily_returns(
                    msr["weights"], test_returns,
                )

            cum_returns = {name: cumulative_wealth(book.returns) for name, book in books.items()}
            caption = (
                "Out-of-sample wealth. Weights were fit on the training window only. "
                "Each sleeve rebalances on the first session of the month and otherwise drifts. "
                f"Cost is {cost_bps:.1f} bps on every dollar bought or sold, including the opening "
                "trade from cash. "
                "Equal weight is 1/N of the selected tickers. "
            )
            if "60/40" in books:
                caption += "60/40 is 60% SPY and 40% AGG (US aggregate bonds), not cash. "
            elif "SPY" not in test_returns.columns:
                caption += "SPY is not in the selected tickers, so there is no market or 60/40 line. "
            caption += (
                "The daily column is the gross weighted sum the mean-variance math describes, with no costs."
            )
            st.caption(caption)
            st.plotly_chart(
                plot_backtest(
                    cum_returns,
                    title="Out-of-sample wealth (monthly rebalance, net of costs)",
                    theme=_THEME,
                ),
                use_container_width=True,
            )
            st.plotly_chart(plot_drawdown(portfolios_dr, theme=_THEME), use_container_width=True)
            if (
                gmvp["weights"] is not None
                and "SPY" in gmvp["weights"].index
                and gmvp["weights"]["SPY"] >= 1.0 - 1e-8
            ):
                st.caption("With no weight cap, minimum variance is 100% SPY, so that line matches SPY.")
            traded = pd.Series({name: book.total_traded for name, book in books.items()}, name="Traded")
            st.caption(
                "Traded is the sum of |Δweight| over the test window (1.00 is the opening trade from cash). "
                + ", ".join(f"{name} {value:.2f}" for name, value in traded.items())
                + "."
            )

            df_metrics, fmt = compare_portfolios(portfolios_dr, risk_free_rate=risk_free)
            st.subheader("Performance Metrics (out-of-sample)")
            st.dataframe(format_comparison(df_metrics, fmt), use_container_width=True)

    # ── Tab 4: Correlations ────────────────────────────────────────────────────
    with tab4:
        fig_corr = plot_correlation_heatmap(data["train_returns"], theme=_THEME)
        st.plotly_chart(fig_corr, use_container_width=True)

else:
    st.info("👈 Configure your parameters in the sidebar and click **Optimise**.")
