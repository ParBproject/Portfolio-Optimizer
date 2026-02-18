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
import numpy as np

from src.data_handler import load_data, simulate_random_portfolios
from src.optimizer    import min_variance, max_sharpe, efficient_frontier
from src.metrics      import (
    portfolio_daily_returns, cumulative_wealth, compare_portfolios
)
from src.visualization import (
    plot_efficient_frontier, plot_weights, plot_backtest,
    plot_drawdown, plot_correlation_heatmap,
)

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

# ── Sidebar inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    tickers_raw = st.text_input(
        "Tickers (comma-separated)",
        value="AAPL, MSFT, GOOGL, AMZN, JPM, SPY",
    )
    tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]

    col1, col2 = st.columns(2)
    start_date = col1.date_input("Start date", value=pd.Timestamp("2015-01-01"))
    end_date   = col2.date_input("End date",   value=pd.Timestamp("2024-12-31"))
    train_end  = st.date_input("Train/test split date", value=pd.Timestamp("2023-12-31"))

    st.subheader("Constraints")
    risk_free = st.slider("Risk-free rate (%)", 0.0, 8.0, 4.0, 0.25) / 100
    max_w = st.slider("Max weight per asset (0 = no cap)", 0.0, 1.0, 0.0, 0.05)
    max_weight = max_w if max_w > 0.0 else None
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
                end=str(end_date),
                train_end=str(train_end),
            )
        except Exception as e:
            st.error(f"Data error: {e}")
            st.stop()

        mu  = data["mu"]
        cov = data["cov"]
        tickers = data["tickers"]

        # ── Optimisation ──────────────────────────────────────────────────────
        gmvp = min_variance(mu, cov, max_weight=max_weight)
        msr  = max_sharpe(mu, cov, risk_free_rate=risk_free, max_weight=max_weight)
        ef   = efficient_frontier(mu, cov, n_points=n_frontier, max_weight=max_weight, risk_free_rate=risk_free)
        rand = simulate_random_portfolios(mu, cov, n_portfolios=n_random, risk_free_rate=risk_free)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🗺️ Efficient Frontier", "⚖️ Portfolios", "📊 Backtest", "🔥 Correlations"]
    )

    # ── Tab 1: Frontier ────────────────────────────────────────────────────────
    with tab1:
        fig = plot_efficient_frontier(ef, rand, gmvp, msr, tickers, risk_free)
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2: Portfolio details ───────────────────────────────────────────────
    with tab2:
        col_g, col_s = st.columns(2)
        with col_g:
            st.subheader("🟢 Global Minimum Variance")
            if gmvp["weights"] is not None:
                st.metric("Return",  f"{gmvp['ret']:.2%}")
                st.metric("Vol",     f"{gmvp['vol']:.2%}")
                st.metric("Sharpe",  f"{gmvp['sharpe']:.3f}")
                st.plotly_chart(plot_weights(gmvp["weights"], "GMVP Weights"), use_container_width=True)
            else:
                st.warning("GMVP could not be solved.")

        with col_s:
            st.subheader("🔴 Maximum Sharpe Ratio")
            if msr["weights"] is not None:
                st.metric("Return",  f"{msr['ret']:.2%}")
                st.metric("Vol",     f"{msr['vol']:.2%}")
                st.metric("Sharpe",  f"{msr['sharpe']:.3f}")
                st.plotly_chart(plot_weights(msr["weights"], "MSR Weights"), use_container_width=True)
            else:
                st.warning("MSR portfolio could not be solved.")

    # ── Tab 3: Backtest ────────────────────────────────────────────────────────
    with tab3:
        test_returns = data["test_returns"]
        if test_returns.empty:
            st.warning("No out-of-sample data. Adjust the train/test split date.")
        else:
            portfolios_dr = {}
            n_assets = len(tickers)

            if gmvp["weights"] is not None:
                dr = portfolio_daily_returns(gmvp["weights"].values, test_returns)
                portfolios_dr["GMVP"] = pd.Series(dr, index=test_returns.index)

            if msr["weights"] is not None:
                dr = portfolio_daily_returns(msr["weights"].values, test_returns)
                portfolios_dr["Max Sharpe"] = pd.Series(dr, index=test_returns.index)

            # Equal-weight benchmark
            eq_w = np.ones(n_assets) / n_assets
            dr   = portfolio_daily_returns(eq_w, test_returns)
            portfolios_dr["Equal Weight"] = pd.Series(dr, index=test_returns.index)

            # Cumulative wealth
            cum_returns = {k: cumulative_wealth(v) for k, v in portfolios_dr.items()}

            st.plotly_chart(plot_backtest(cum_returns), use_container_width=True)
            st.plotly_chart(plot_drawdown(portfolios_dr), use_container_width=True)

            # Metrics table
            df_metrics, fmt = compare_portfolios(portfolios_dr, risk_free_rate=risk_free)
            styled = df_metrics.style
            for metric, f in fmt.items():
                styled = styled.format({col: f for col in df_metrics.columns}, subset=pd.IndexSlice[[metric], :])
            st.subheader("Performance Metrics (out-of-sample)")
            st.dataframe(styled, use_container_width=True)

    # ── Tab 4: Correlations ────────────────────────────────────────────────────
    with tab4:
        fig_corr = plot_correlation_heatmap(data["train_returns"])
        st.plotly_chart(fig_corr, use_container_width=True)

else:
    st.info("👈 Configure your parameters in the sidebar and click **Optimise**.")
