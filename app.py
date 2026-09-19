"""Interactive quantitative portfolio research dashboard."""

from __future__ import annotations

import os
import sys
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from src.backtest import backtest_fixed_allocation, walk_forward_backtest
from src.data_handler import load_data, shrink_covariance, simulate_random_portfolios
from src.metrics import compare_portfolios, cumulative_wealth
from src.optimizer import efficient_frontier, max_sharpe, min_variance
from src.visualization import (
    plot_backtest,
    plot_correlation_heatmap,
    plot_drawdown,
    plot_efficient_frontier,
    plot_weights,
)


st.set_page_config(
    page_title="Portfolio Research Lab",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background: #F8FAFC;
    }
    .block-container {
        max-width: 1450px;
        padding-top: 1.4rem;
        padding-bottom: 3rem;
    }
    .hero {
        padding: 2.0rem 2.2rem;
        border-radius: 22px;
        background:
            radial-gradient(circle at 85% 10%, rgba(56,189,248,.22), transparent 30%),
            linear-gradient(135deg, #0F172A 0%, #102A43 58%, #0F766E 130%);
        color: white;
        box-shadow: 0 18px 50px rgba(15, 23, 42, .15);
        margin-bottom: 1.1rem;
    }
    .hero-kicker {
        font-size: .78rem;
        letter-spacing: .16em;
        text-transform: uppercase;
        color: #99F6E4;
        font-weight: 700;
        margin-bottom: .55rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2.25rem;
        line-height: 1.08;
        letter-spacing: -.03em;
    }
    .hero p {
        margin: .75rem 0 0 0;
        max-width: 850px;
        color: #D7E3F1;
        font-size: 1.02rem;
        line-height: 1.6;
    }
    .signal-card {
        padding: 1rem 1.05rem;
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        min-height: 104px;
        box-shadow: 0 5px 18px rgba(15, 23, 42, .04);
    }
    .signal-label {
        color: #64748B;
        font-size: .76rem;
        text-transform: uppercase;
        letter-spacing: .09em;
        font-weight: 700;
    }
    .signal-value {
        color: #0F172A;
        font-size: 1.48rem;
        font-weight: 750;
        margin-top: .28rem;
    }
    .signal-note {
        color: #64748B;
        font-size: .79rem;
        margin-top: .18rem;
    }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 15px;
        padding: .85rem 1rem;
        box-shadow: 0 4px 14px rgba(15, 23, 42, .035);
    }
    div[data-testid="stMetricLabel"] {
        color: #64748B;
    }
    div[data-testid="stMetricValue"] {
        color: #0F172A;
    }
    section[data-testid="stSidebar"] {
        background: #F1F5F9;
        border-right: 1px solid #E2E8F0;
    }
    .method-box {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 18px;
        padding: 1.1rem 1.25rem;
        margin-bottom: .8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def signal_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="signal-card">
          <div class="signal-label">{label}</div>
          <div class="signal-value">{value}</div>
          <div class="signal-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">Quantitative Portfolio Research</div>
      <h1>Portfolio Research Lab</h1>
      <p>
        A reproducible research environment for mean-variance optimization,
        covariance-aware allocation, no-lookahead walk-forward testing,
        implementation costs, tail-risk diagnostics, and benchmark comparison.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown("### Research configuration")
    st.caption("Market universe, estimation assumptions, and implementation constraints.")

    tickers_raw = st.text_input(
        "Asset universe",
        value="AAPL, MSFT, GOOGL, AMZN, JPM, SPY",
        help="Comma-separated Yahoo Finance ticker symbols.",
    )
    tickers = [ticker.strip().upper() for ticker in tickers_raw.split(",") if ticker.strip()]

    start_date = st.date_input("History start", value=date(2016, 1, 1))
    end_date = st.date_input("History end", value=date.today())
    train_end = st.date_input("Static train/test split", value=date(2023, 12, 29))

    st.divider()
    st.markdown("#### Optimization")
    risk_free = st.slider("Risk-free rate", 0.0, 8.0, 4.0, 0.25) / 100.0
    max_weight_raw = st.slider(
        "Maximum asset weight",
        0.0,
        1.0,
        0.45,
        0.05,
        help="Set to 1.0 for no effective concentration cap.",
    )
    max_weight = None if max_weight_raw >= 1.0 else max_weight_raw
    covariance_shrinkage = st.slider(
        "Covariance shrinkage",
        0.0,
        0.80,
        0.15,
        0.05,
        help="Shrinks off-diagonal sample covariance toward zero.",
    )
    n_frontier = st.slider("Frontier resolution", 20, 100, 60, 5)
    n_random = st.slider("Opportunity-set simulations", 1000, 15000, 6000, 500)

    st.divider()
    st.markdown("#### Walk-forward implementation")
    lookback_days = st.select_slider(
        "Estimation lookback",
        options=[63, 126, 189, 252, 378, 504],
        value=252,
        format_func=lambda value: f"{value} trading days",
    )
    rebalance_every = st.select_slider(
        "Rebalance frequency",
        options=[5, 10, 21, 42, 63],
        value=21,
        format_func=lambda value: f"Every {value} days",
    )
    transaction_cost_bps = st.slider(
        "Transaction cost",
        0.0,
        50.0,
        5.0,
        1.0,
        help="One-way proportional implementation cost in basis points.",
    )

    run_btn = st.button(
        "Run quantitative analysis",
        type="primary",
        use_container_width=True,
    )


if not run_btn:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        signal_card("Optimization", "CVXPY", "Constrained quadratic programming")
    with c2:
        signal_card("Validation", "Walk-forward", "No-lookahead rolling estimation")
    with c3:
        signal_card("Risk", "8 metrics", "Sharpe, Sortino, VaR, ES & drawdown")
    with c4:
        signal_card("Implementation", "Cost-aware", "Turnover and trading friction")

    st.markdown("### What this project demonstrates")
    left, right = st.columns(2)
    with left:
        st.markdown(
            """
            <div class="method-box">
            <b>Quantitative research</b><br><br>
            Efficient-frontier construction, minimum-variance and maximum-Sharpe
            optimization, sample-covariance shrinkage, rolling estimation, and
            explicit portfolio constraints.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div class="method-box">
            <b>Data & model discipline</b><br><br>
            Historical market-data ingestion, train/test separation, no-lookahead
            walk-forward testing, benchmark comparison, downside-risk metrics,
            transaction costs, and reproducible random simulations.
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.stop()


if not tickers:
    st.error("Enter at least one ticker.")
    st.stop()
if start_date >= end_date:
    st.error("History start must be before history end.")
    st.stop()
if train_end <= start_date or train_end >= end_date:
    st.error("The static train/test split must fall inside the selected history.")
    st.stop()
if max_weight is not None and max_weight * len(tickers) < 1.0 - 1e-9:
    st.error(
        "The maximum-weight constraint is infeasible for this number of assets. "
        "Increase the cap or add more assets."
    )
    st.stop()


with st.spinner("Loading market history and solving portfolio models..."):
    try:
        data = load_data(
            tickers,
            start=str(start_date),
            end=str(end_date),
            train_end=str(train_end),
        )
        mu = data["mu"]
        sample_cov = data["cov"]
        cov = shrink_covariance(sample_cov, covariance_shrinkage)
        tickers = data["tickers"]

        gmvp = min_variance(
            mu,
            cov,
            max_weight=max_weight,
            risk_free_rate=risk_free,
        )
        msr = max_sharpe(
            mu,
            cov,
            risk_free_rate=risk_free,
            max_weight=max_weight,
        )
        frontier = efficient_frontier(
            mu,
            cov,
            n_points=n_frontier,
            max_weight=max_weight,
            risk_free_rate=risk_free,
        )
        random_portfolios = simulate_random_portfolios(
            mu,
            cov,
            n_portfolios=n_random,
            risk_free_rate=risk_free,
        )
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
        st.stop()


header1, header2, header3, header4 = st.columns(4)
with header1:
    signal_card("Assets", str(len(tickers)), " / ".join(tickers[:4]) + ("…" if len(tickers) > 4 else ""))
with header2:
    signal_card(
        "In-sample observations",
        f"{len(data['train_returns']):,}",
        f"Through {pd.Timestamp(train_end).date()}",
    )
with header3:
    signal_card(
        "Out-of-sample observations",
        f"{len(data['test_returns']):,}",
        "Static holdout window",
    )
with header4:
    signal_card(
        "Covariance estimator",
        f"{covariance_shrinkage:.0%} shrinkage",
        "Diagonal-target regularization",
    )

st.write("")

frontier_tab, allocation_tab, research_tab, correlation_tab, methodology_tab = st.tabs(
    [
        "Efficient Frontier",
        "Portfolio Allocations",
        "Walk-Forward Research",
        "Dependence Structure",
        "Methodology",
    ]
)


with frontier_tab:
    st.plotly_chart(
        plot_efficient_frontier(
            frontier,
            random_portfolios,
            gmvp,
            msr,
            tickers,
            risk_free,
        ),
        use_container_width=True,
    )

    if gmvp.get("weights") is not None and msr.get("weights") is not None:
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Min-var return", f"{gmvp['ret']:.2%}")
        m2.metric("Min-var volatility", f"{gmvp['vol']:.2%}")
        m3.metric("Min-var Sharpe", f"{gmvp['sharpe']:.2f}")
        m4.metric("Max-Sharpe return", f"{msr['ret']:.2%}")
        m5.metric("Max-Sharpe volatility", f"{msr['vol']:.2%}")
        m6.metric("Max-Sharpe ratio", f"{msr['sharpe']:.2f}")


with allocation_tab:
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("#### Global Minimum Variance")
        if gmvp.get("weights") is not None:
            st.plotly_chart(
                plot_weights(gmvp["weights"], "Minimum-Variance Allocation"),
                use_container_width=True,
            )
            st.dataframe(
                gmvp["weights"]
                .sort_values(ascending=False)
                .rename("Weight")
                .to_frame()
                .style.format("{:.2%}"),
                use_container_width=True,
            )
        else:
            st.warning("Minimum-variance optimization was infeasible.")

    with col_right:
        st.markdown("#### Maximum Sharpe")
        if msr.get("weights") is not None:
            st.plotly_chart(
                plot_weights(msr["weights"], "Maximum-Sharpe Allocation"),
                use_container_width=True,
            )
            st.dataframe(
                msr["weights"]
                .sort_values(ascending=False)
                .rename("Weight")
                .to_frame()
                .style.format("{:.2%}"),
                use_container_width=True,
            )
        else:
            st.warning("Maximum-Sharpe optimization was infeasible.")


with research_tab:
    st.markdown("### No-lookahead walk-forward evaluation")
    st.caption(
        "Each rebalance estimates expected returns and covariance using only the "
        "preceding lookback window. Portfolio weights are then held until the next "
        "rebalance, with explicit turnover-based transaction costs."
    )

    try:
        wf_msr = walk_forward_backtest(
            data["prices"],
            strategy="max_sharpe",
            lookback_days=int(lookback_days),
            rebalance_every=int(rebalance_every),
            risk_free_rate=risk_free,
            max_weight=max_weight,
            transaction_cost_bps=transaction_cost_bps,
        )
        wf_gmvp = walk_forward_backtest(
            data["prices"],
            strategy="min_variance",
            lookback_days=int(lookback_days),
            rebalance_every=int(rebalance_every),
            risk_free_rate=risk_free,
            max_weight=max_weight,
            transaction_cost_bps=transaction_cost_bps,
        )

        common_start = max(wf_msr.returns.index.min(), wf_gmvp.returns.index.min())
        equal_prices = data["prices"].loc[common_start:]
        equal_weight = np.ones(len(tickers)) / len(tickers)
        equal = backtest_fixed_allocation(
            equal_prices,
            equal_weight,
            rebalance_every=int(rebalance_every),
            transaction_cost_bps=transaction_cost_bps,
        )

        simple_returns = pd.concat(
            {
                "Walk-Forward Max Sharpe": wf_msr.returns,
                "Walk-Forward Min Variance": wf_gmvp.returns,
                "Equal Weight": equal.returns,
            },
            axis=1,
            join="inner",
        ).dropna()
        log_returns = {
            label: np.log1p(simple_returns[label])
            for label in simple_returns.columns
        }
        wealth = {
            label: cumulative_wealth(series)
            for label, series in log_returns.items()
        }

        st.plotly_chart(plot_backtest(wealth), use_container_width=True)
        st.plotly_chart(plot_drawdown(log_returns), use_container_width=True)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Max-Sharpe turnover", f"{wf_msr.total_turnover:.2f}×")
        k2.metric("Max-Sharpe modeled costs", f"{wf_msr.total_transaction_cost:.2%}")
        k3.metric("Min-var turnover", f"{wf_gmvp.total_turnover:.2f}×")
        k4.metric("Min-var modeled costs", f"{wf_gmvp.total_transaction_cost:.2%}")

        metrics, formats = compare_portfolios(
            log_returns,
            risk_free_rate=risk_free,
        )
        styled = metrics.style
        for metric, format_string in formats.items():
            styled = styled.format(
                {column: format_string for column in metrics.columns},
                subset=pd.IndexSlice[[metric], :],
            )
        st.markdown("#### Risk-adjusted performance")
        st.dataframe(styled, use_container_width=True)

        st.markdown("#### Allocation path — walk-forward maximum Sharpe")
        st.area_chart(wf_msr.weights)
    except Exception as exc:
        st.warning(f"Walk-forward analysis could not be completed: {exc}")


with correlation_tab:
    st.plotly_chart(
        plot_correlation_heatmap(data["train_returns"]),
        use_container_width=True,
    )
    st.markdown("#### Annualized covariance used by the optimizer")
    st.dataframe(
        pd.DataFrame(cov, index=tickers, columns=tickers).style.format("{:.4f}"),
        use_container_width=True,
    )


with methodology_tab:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div class="method-box">
            <b>Optimization problem</b><br><br>
            Minimize portfolio variance <b>wᵀΣw</b>, subject to full investment,
            long-only weights, an optional concentration cap, and return targets
            along the efficient frontier.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="method-box">
            <b>Robust covariance option</b><br><br>
            The sample covariance matrix can be shrunk toward its diagonal.
            This reduces the influence of unstable cross-asset covariance estimates
            without hiding the underlying sample estimator.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="method-box">
            <b>No-lookahead validation</b><br><br>
            Walk-forward weights at each rebalance are estimated only from the
            preceding window. Future observations never enter the estimation set.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="method-box">
            <b>Implementation realism</b><br><br>
            Rebalancing creates turnover. The backtest applies a configurable
            basis-point cost to one-way turnover and reports the cumulative modeled
            implementation cost alongside performance.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Research limitations")
    st.markdown(
        """
        Historical estimates are regime-dependent. The model does not include taxes,
        market impact, borrow costs, execution latency, or every institutional
        constraint. Backtested results are research evidence, not a forecast.
        """
    )

st.caption(
    "Educational quantitative-finance portfolio project. Historical and simulated "
    "results do not constitute investment advice."
)
