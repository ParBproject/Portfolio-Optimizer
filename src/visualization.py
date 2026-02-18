"""
visualization.py
----------------
Interactive Plotly visualisations for the Efficient Frontier and backtesting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ── Colour palette ─────────────────────────────────────────────────────────────
GMVP_COLOR   = "#00CC66"   # green
MSR_COLOR    = "#FF4136"   # red
FRONTIER_CLR = "#003366"   # dark blue
BG_COLOR     = "#F8F9FA"


def plot_efficient_frontier(
    frontier_df: pd.DataFrame,
    random_df: pd.DataFrame | None = None,
    gmvp: dict | None = None,
    msr: dict | None = None,
    tickers: list[str] | None = None,
    risk_free_rate: float = 0.04,
    title: str = "Efficient Frontier",
) -> go.Figure:
    """
    Interactive Plotly scatter of the Efficient Frontier.

    Parameters
    ----------
    frontier_df  : DataFrame from optimizer.efficient_frontier()
                   columns: [ret, vol, sharpe, <tickers…>]
    random_df    : DataFrame from data_handler.simulate_random_portfolios()
                   (optional – shows background cloud)
    gmvp         : dict from optimizer.min_variance() – Global Min Variance
    msr          : dict from optimizer.max_sharpe()   – Max Sharpe Ratio
    tickers      : list of asset names (for hover labels)
    risk_free_rate: for Capital Market Line
    title        : figure title

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig = go.Figure()

    # ── Random portfolios (background cloud) ───────────────────────────────────
    if random_df is not None and len(random_df):
        t = tickers or [c for c in random_df.columns if c not in ("ret", "vol", "sharpe")]
        hover_txt = _build_hover(random_df, t)
        fig.add_trace(go.Scatter(
            x=random_df["vol"], y=random_df["ret"],
            mode="markers",
            marker=dict(
                size=4,
                color=random_df["sharpe"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Sharpe", thickness=14),
                opacity=0.55,
            ),
            text=hover_txt,
            hovertemplate="%{text}<extra>Random Portfolio</extra>",
            name="Random Portfolios",
        ))

    # ── Efficient Frontier curve ───────────────────────────────────────────────
    t = tickers or [c for c in frontier_df.columns if c not in ("ret", "vol", "sharpe")]
    hover_ef = _build_hover(frontier_df, t)
    fig.add_trace(go.Scatter(
        x=frontier_df["vol"], y=frontier_df["ret"],
        mode="lines+markers",
        line=dict(color=FRONTIER_CLR, width=3),
        marker=dict(size=5, color=FRONTIER_CLR),
        text=hover_ef,
        hovertemplate="%{text}<extra>Efficient Frontier</extra>",
        name="Efficient Frontier",
    ))

    # ── Capital Market Line ────────────────────────────────────────────────────
    if msr is not None and msr.get("weights") is not None:
        x_cml = np.array([0, frontier_df["vol"].max() * 1.15])
        slope  = (msr["ret"] - risk_free_rate) / msr["vol"]
        y_cml  = risk_free_rate + slope * x_cml
        fig.add_trace(go.Scatter(
            x=x_cml, y=y_cml,
            mode="lines",
            line=dict(color=MSR_COLOR, dash="dash", width=1.5),
            name="Capital Market Line",
        ))

    # ── GMVP marker ───────────────────────────────────────────────────────────
    if gmvp is not None and gmvp.get("weights") is not None:
        w = gmvp["weights"]
        hover = (
            f"<b>Global Min Variance</b><br>"
            f"Return: {gmvp['ret']:.2%}  Vol: {gmvp['vol']:.2%}  Sharpe: {gmvp['sharpe']:.3f}<br>"
            + "<br>".join(f"{k}: {v:.2%}" for k, v in w.items())
        )
        fig.add_trace(go.Scatter(
            x=[gmvp["vol"]], y=[gmvp["ret"]],
            mode="markers",
            marker=dict(size=18, color=GMVP_COLOR, symbol="diamond", line=dict(width=2, color="white")),
            name="Min Variance",
            hovertemplate=hover + "<extra></extra>",
        ))

    # ── MSR marker ────────────────────────────────────────────────────────────
    if msr is not None and msr.get("weights") is not None:
        w = msr["weights"]
        hover = (
            f"<b>Max Sharpe Ratio</b><br>"
            f"Return: {msr['ret']:.2%}  Vol: {msr['vol']:.2%}  Sharpe: {msr['sharpe']:.3f}<br>"
            + "<br>".join(f"{k}: {v:.2%}" for k, v in w.items())
        )
        fig.add_trace(go.Scatter(
            x=[msr["vol"]], y=[msr["ret"]],
            mode="markers",
            marker=dict(size=22, color=MSR_COLOR, symbol="star", line=dict(width=2, color="white")),
            name="Max Sharpe",
            hovertemplate=hover + "<extra></extra>",
        ))

    # ── Risk-free rate annotation ──────────────────────────────────────────────
    fig.add_annotation(
        x=0, y=risk_free_rate, text=f"Rf={risk_free_rate:.1%}",
        showarrow=False, font=dict(size=11, color="grey"),
        xanchor="left", yanchor="bottom",
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=22)),
        xaxis=dict(title="Annualised Volatility (σ)", tickformat=".1%", gridcolor="#E8E8E8"),
        yaxis=dict(title="Annualised Expected Return (μ)", tickformat=".1%", gridcolor="#E8E8E8"),
        plot_bgcolor=BG_COLOR, paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.15, x=0.0),
        hovermode="closest",
        width=950, height=620,
    )
    return fig


def plot_weights(weights: pd.Series, title: str = "Portfolio Weights") -> go.Figure:
    """Bar chart of portfolio weights."""
    sorted_w = weights.sort_values(ascending=False)
    colors = px.colors.qualitative.Bold[:len(sorted_w)]
    fig = go.Figure(go.Bar(
        x=sorted_w.index, y=sorted_w.values,
        marker_color=colors,
        text=[f"{v:.1%}" for v in sorted_w.values],
        textposition="outside",
    ))
    fig.update_layout(
        title=title, yaxis=dict(title="Weight", tickformat=".0%", range=[0, sorted_w.max() * 1.2]),
        plot_bgcolor=BG_COLOR, width=700, height=420,
    )
    return fig


def plot_backtest(
    cum_returns: dict[str, pd.Series],
    title: str = "Out-of-Sample Cumulative Returns",
) -> go.Figure:
    """
    Overlay multiple cumulative return series.

    Parameters
    ----------
    cum_returns : {label: cumulative_wealth_series}
    """
    palette = px.colors.qualitative.Safe
    fig = go.Figure()
    for i, (label, series) in enumerate(cum_returns.items()):
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values,
            mode="lines", name=label,
            line=dict(width=2.5, color=palette[i % len(palette)]),
            hovertemplate=f"<b>{label}</b><br>Date: %{{x|%Y-%m-%d}}<br>Value: %{{y:.3f}}<extra></extra>",
        ))
    fig.add_hline(y=1.0, line_dash="dot", line_color="grey", annotation_text="Start")
    fig.update_layout(
        title=title,
        xaxis=dict(title="Date", gridcolor="#E8E8E8"),
        yaxis=dict(title="Cumulative Wealth ($1 start)", gridcolor="#E8E8E8"),
        plot_bgcolor=BG_COLOR, legend=dict(orientation="h", y=-0.15),
        width=950, height=520,
    )
    return fig


def plot_drawdown(
    daily_returns_dict: dict[str, pd.Series],
    title: str = "Drawdown",
) -> go.Figure:
    """Drawdown chart for one or more portfolios."""
    from src.metrics import cumulative_wealth, max_drawdown
    palette = px.colors.qualitative.Safe
    fig = go.Figure()
    for i, (label, dr) in enumerate(daily_returns_dict.items()):
        cum  = cumulative_wealth(dr)
        peak = cum.cummax()
        dd   = (cum - peak) / peak
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values,
            mode="lines", name=label, fill="tozeroy",
            line=dict(color=palette[i % len(palette)], width=1.5),
        ))
    fig.update_layout(
        title=title,
        xaxis=dict(title="Date", gridcolor="#E8E8E8"),
        yaxis=dict(title="Drawdown", tickformat=".1%", gridcolor="#E8E8E8"),
        plot_bgcolor=BG_COLOR, width=950, height=380,
    )
    return fig


def plot_correlation_heatmap(returns: pd.DataFrame, title: str = "Asset Correlation Matrix") -> go.Figure:
    """Annotated correlation heatmap."""
    corr = returns.corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale="RdBu", zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}", textfont=dict(size=12),
        hoverongaps=False,
    ))
    fig.update_layout(title=title, width=600, height=550)
    return fig


# ── Private helpers ────────────────────────────────────────────────────────────

def _build_hover(df: pd.DataFrame, tickers: list[str]) -> list[str]:
    """Build hover-text strings for scatter traces."""
    texts = []
    for _, row in df.iterrows():
        lines = [
            f"Return: {row['ret']:.2%}",
            f"Volatility: {row['vol']:.2%}",
            f"Sharpe: {row['sharpe']:.3f}",
        ]
        for t in tickers:
            if t in row:
                lines.append(f"{t}: {row[t]:.2%}")
        texts.append("<br>".join(lines))
    return texts
