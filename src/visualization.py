"""Professional Plotly visualisations for portfolio research."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

NAVY = "#0F172A"
TEAL = "#14B8A6"
CYAN = "#38BDF8"
BLUE = "#2563EB"
ROSE = "#F43F5E"
SLATE = "#64748B"
GRID = "#E2E8F0"
PAPER = "#FFFFFF"


def _base_layout(title: str, height: int = 520) -> dict:
    return {
        "title": {
            "text": title,
            "font": {"size": 22, "color": NAVY},
            "x": 0.02,
            "xanchor": "left",
        },
        "height": height,
        "paper_bgcolor": PAPER,
        "plot_bgcolor": PAPER,
        "font": {"family": "Inter, Arial, sans-serif", "color": NAVY},
        "margin": {"l": 60, "r": 30, "t": 70, "b": 70},
        "legend": {
            "orientation": "h",
            "y": -0.16,
            "x": 0.0,
            "font": {"size": 12},
        },
        "hovermode": "closest",
    }


def plot_efficient_frontier(
    frontier_df: pd.DataFrame,
    random_df: pd.DataFrame | None = None,
    gmvp: dict | None = None,
    msr: dict | None = None,
    tickers: list[str] | None = None,
    risk_free_rate: float = 0.04,
    title: str = "Efficient Frontier & Opportunity Set",
) -> go.Figure:
    """Interactive efficient-frontier chart with portfolio diagnostics."""
    fig = go.Figure()

    if random_df is not None and len(random_df):
        labels = tickers or [
            column
            for column in random_df.columns
            if column not in ("ret", "vol", "sharpe")
        ]
        fig.add_trace(
            go.Scatter(
                x=random_df["vol"],
                y=random_df["ret"],
                mode="markers",
                marker={
                    "size": 5,
                    "color": random_df["sharpe"],
                    "colorscale": "Tealgrn",
                    "showscale": True,
                    "colorbar": {
                        "title": "Sharpe",
                        "thickness": 12,
                        "outlinewidth": 0,
                    },
                    "opacity": 0.38,
                    "line": {"width": 0},
                },
                text=_build_hover(random_df, labels),
                hovertemplate="%{text}<extra>Simulated portfolio</extra>",
                name="Opportunity set",
            )
        )

    labels = tickers or [
        column
        for column in frontier_df.columns
        if column not in ("ret", "vol", "sharpe")
    ]
    fig.add_trace(
        go.Scatter(
            x=frontier_df["vol"],
            y=frontier_df["ret"],
            mode="lines",
            line={"color": NAVY, "width": 4},
            text=_build_hover(frontier_df, labels),
            hovertemplate="%{text}<extra>Efficient frontier</extra>",
            name="Efficient frontier",
        )
    )

    if msr is not None and msr.get("weights") is not None and len(frontier_df):
        x_cml = np.array([0.0, float(frontier_df["vol"].max()) * 1.12])
        slope = (msr["ret"] - risk_free_rate) / msr["vol"]
        y_cml = risk_free_rate + slope * x_cml
        fig.add_trace(
            go.Scatter(
                x=x_cml,
                y=y_cml,
                mode="lines",
                line={"color": CYAN, "dash": "dash", "width": 2},
                name="Capital market line",
                hoverinfo="skip",
            )
        )

    if gmvp is not None and gmvp.get("weights") is not None:
        hover = _portfolio_hover("Minimum Variance", gmvp)
        fig.add_trace(
            go.Scatter(
                x=[gmvp["vol"]],
                y=[gmvp["ret"]],
                mode="markers",
                marker={
                    "size": 17,
                    "color": TEAL,
                    "symbol": "diamond",
                    "line": {"width": 2, "color": PAPER},
                },
                name="Minimum variance",
                hovertemplate=hover + "<extra></extra>",
            )
        )

    if msr is not None and msr.get("weights") is not None:
        hover = _portfolio_hover("Maximum Sharpe", msr)
        fig.add_trace(
            go.Scatter(
                x=[msr["vol"]],
                y=[msr["ret"]],
                mode="markers",
                marker={
                    "size": 20,
                    "color": ROSE,
                    "symbol": "star",
                    "line": {"width": 2, "color": PAPER},
                },
                name="Maximum Sharpe",
                hovertemplate=hover + "<extra></extra>",
            )
        )

    fig.add_annotation(
        x=0,
        y=risk_free_rate,
        text=f"Risk-free rate {risk_free_rate:.1%}",
        showarrow=False,
        font={"size": 11, "color": SLATE},
        xanchor="left",
        yanchor="bottom",
    )
    layout = _base_layout(title, height=610)
    layout["xaxis"] = {
        "title": "Annualised volatility",
        "tickformat": ".1%",
        "gridcolor": GRID,
        "zeroline": False,
    }
    layout["yaxis"] = {
        "title": "Annualised expected return",
        "tickformat": ".1%",
        "gridcolor": GRID,
        "zeroline": False,
    }
    fig.update_layout(**layout)
    return fig


def plot_weights(weights: pd.Series, title: str = "Portfolio Allocation") -> go.Figure:
    """Horizontal allocation chart ordered by portfolio weight."""
    sorted_weights = weights.sort_values(ascending=True)
    fig = go.Figure(
        go.Bar(
            x=sorted_weights.values,
            y=sorted_weights.index,
            orientation="h",
            marker={
                "color": sorted_weights.values,
                "colorscale": [[0, "#CCFBF1"], [1, TEAL]],
                "line": {"width": 0},
            },
            text=[f"{value:.1%}" for value in sorted_weights.values],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Weight: %{x:.2%}<extra></extra>",
        )
    )
    layout = _base_layout(title, height=max(360, 70 + len(sorted_weights) * 45))
    layout["xaxis"] = {
        "title": "Portfolio weight",
        "tickformat": ".0%",
        "gridcolor": GRID,
        "range": [0, max(float(sorted_weights.max()) * 1.2, 0.1)],
    }
    layout["yaxis"] = {"title": ""}
    layout["showlegend"] = False
    fig.update_layout(**layout)
    return fig


def plot_backtest(
    cum_returns: dict[str, pd.Series],
    title: str = "Out-of-Sample Wealth",
) -> go.Figure:
    """Overlay cumulative wealth series."""
    colors = [NAVY, TEAL, CYAN, ROSE, BLUE]
    fig = go.Figure()
    for index, (label, series) in enumerate(cum_returns.items()):
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                name=label,
                line={"width": 2.7, "color": colors[index % len(colors)]},
                hovertemplate=(
                    f"<b>{label}</b><br>Date: %{{x|%Y-%m-%d}}"
                    "<br>Wealth: %{y:.3f}<extra></extra>"
                ),
            )
        )
    fig.add_hline(y=1.0, line_dash="dot", line_color=SLATE, opacity=0.5)
    layout = _base_layout(title, height=500)
    layout["xaxis"] = {"title": "", "gridcolor": GRID}
    layout["yaxis"] = {"title": "Wealth index", "gridcolor": GRID}
    fig.update_layout(**layout)
    return fig


def plot_drawdown(
    daily_returns_dict: dict[str, pd.Series],
    title: str = "Drawdown Profile",
) -> go.Figure:
    """Drawdown chart for one or more portfolios."""
    from src.metrics import cumulative_wealth

    colors = [NAVY, TEAL, CYAN, ROSE, BLUE]
    fig = go.Figure()
    for index, (label, daily_returns) in enumerate(daily_returns_dict.items()):
        wealth = cumulative_wealth(daily_returns)
        peak = wealth.cummax()
        drawdown = wealth / peak - 1.0
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                mode="lines",
                name=label,
                line={"color": colors[index % len(colors)], "width": 2},
                hovertemplate=(
                    f"<b>{label}</b><br>Date: %{{x|%Y-%m-%d}}"
                    "<br>Drawdown: %{y:.2%}<extra></extra>"
                ),
            )
        )
    layout = _base_layout(title, height=370)
    layout["xaxis"] = {"title": "", "gridcolor": GRID}
    layout["yaxis"] = {
        "title": "Drawdown",
        "tickformat": ".1%",
        "gridcolor": GRID,
    }
    fig.update_layout(**layout)
    return fig


def plot_correlation_heatmap(
    returns: pd.DataFrame,
    title: str = "Asset Correlation Matrix",
) -> go.Figure:
    """Annotated correlation heatmap."""
    correlation = returns.corr()
    fig = go.Figure(
        go.Heatmap(
            z=correlation.values,
            x=correlation.columns,
            y=correlation.index,
            colorscale=[
                [0.0, "#DBEAFE"],
                [0.5, "#F8FAFC"],
                [1.0, "#0F766E"],
            ],
            zmin=-1,
            zmax=1,
            text=np.round(correlation.values, 2),
            texttemplate="%{text}",
            textfont={"size": 12},
            hovertemplate="%{y} / %{x}<br>Correlation: %{z:.2f}<extra></extra>",
            colorbar={"title": "ρ", "thickness": 12, "outlinewidth": 0},
        )
    )
    layout = _base_layout(title, height=540)
    layout["xaxis"] = {"side": "bottom"}
    layout["yaxis"] = {"autorange": "reversed"}
    fig.update_layout(**layout)
    return fig


def _portfolio_hover(label: str, portfolio: dict) -> str:
    weights = portfolio["weights"]
    allocation = "<br>".join(
        f"{ticker}: {weight:.1%}"
        for ticker, weight in weights.items()
        if weight > 0.005
    )
    return (
        f"<b>{label}</b><br>"
        f"Return: {portfolio['ret']:.2%}<br>"
        f"Volatility: {portfolio['vol']:.2%}<br>"
        f"Sharpe: {portfolio['sharpe']:.2f}<br><br>{allocation}"
    )


def _build_hover(df: pd.DataFrame, tickers: list[str]) -> list[str]:
    texts: list[str] = []
    for _, row in df.iterrows():
        lines = [
            f"Return: {row['ret']:.2%}",
            f"Volatility: {row['vol']:.2%}",
            f"Sharpe: {row['sharpe']:.2f}",
        ]
        for ticker in tickers:
            if ticker in row and row[ticker] > 0.005:
                lines.append(f"{ticker}: {row[ticker]:.1%}")
        texts.append("<br>".join(lines))
    return texts
