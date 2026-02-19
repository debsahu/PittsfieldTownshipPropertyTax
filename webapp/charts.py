"""Plotly chart factory functions for property tax analysis."""

import plotly.graph_objects as go
import pandas as pd
import numpy as np

NAVY = "#003366"
LIGHT_BLUE = "#E8F0FE"
RED = "#C0392B"
GREEN = "#27AE60"
ORANGE = "#E67E22"
GRAY = "#7F8C8D"


def ecf_trend_chart(ecf_trend: dict[int, float | None]) -> go.Figure:
    """Line chart showing ECF trend with a dashed line at 1.0."""
    years = sorted(ecf_trend.keys())
    values = [ecf_trend[y] for y in years]
    valid_years = [y for y, v in zip(years, values) if v is not None]
    valid_values = [v for v in values if v is not None]

    if not valid_values:
        fig = go.Figure()
        fig.add_annotation(text="No ECF data available", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        return fig

    colors = [RED if v < 1.0 else GREEN for v in valid_values]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=valid_years, y=valid_values, mode="lines+markers+text",
        line=dict(color=NAVY, width=3), marker=dict(size=12, color=colors),
        text=[f"{v:.3f}" for v in valid_values],
        textposition="top center", textfont=dict(size=14, color=NAVY),
        hovertemplate="Year: %{x}<br>ECF: %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=1.0, line_dash="dash", line_color=GRAY,
                  annotation_text="Fair Assessment (1.0)", annotation_position="bottom right")

    fig.update_layout(
        title="Area ECF Trend (2024-2026)", xaxis_title="Year", yaxis_title="ECF",
        xaxis=dict(tickmode="array", tickvals=years, dtick=1),
        yaxis=dict(range=[min(0.5, min(valid_values) - 0.1), max(1.5, max(valid_values) + 0.1)]),
        height=400, template="plotly_white",
    )
    return fig


def ecf_distribution_chart(ecf_properties: list[dict], user_ecf: float | None = None) -> go.Figure:
    """Box plot showing ECF distribution per year."""
    if not ecf_properties:
        fig = go.Figure()
        fig.add_annotation(text="No individual ECF data", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        return fig

    df = pd.DataFrame(ecf_properties)
    fig = go.Figure()
    for year in sorted(df["year"].unique()):
        year_data = df[df["year"] == year]["ecf"]
        fig.add_trace(go.Box(
            y=year_data, name=str(year), boxpoints="all", jitter=0.3, pointpos=-1.5,
            marker=dict(size=6), hovertext=df[df["year"] == year]["address"],
        ))

    fig.add_hline(y=1.0, line_dash="dash", line_color=GRAY,
                  annotation_text="Fair Assessment", annotation_position="bottom right")

    fig.update_layout(
        title="Individual Property ECF Distribution",
        yaxis_title="ECF", height=400, template="plotly_white", showlegend=False,
    )
    return fig


def sales_scatter_chart(sales_df: pd.DataFrame, user_tcv: float) -> go.Figure:
    """Scatter chart of sales over time with TCV line."""
    if sales_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No comparable sales found", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        return fig

    df = sales_df.copy()
    df["Sale_Date_Parsed"] = pd.to_datetime(df["Sale_Date"], format="mixed", errors="coerce")
    df = df.dropna(subset=["Sale_Date_Parsed", "Adj_Sale"])

    colors = [GREEN if p >= user_tcv else RED for p in df["Adj_Sale"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Sale_Date_Parsed"], y=df["Adj_Sale"], mode="markers",
        marker=dict(size=10, color=colors, line=dict(width=1, color="white")),
        text=df["Street_Address"],
        hovertemplate="<b>%{text}</b><br>Date: %{x|%b %d, %Y}<br>Price: $%{y:,.0f}<extra></extra>",
    ))
    fig.add_hline(y=user_tcv, line_dash="dash", line_color=NAVY, line_width=2,
                  annotation_text=f"Your TCV: ${user_tcv:,.0f}", annotation_position="top left",
                  annotation_font=dict(color=NAVY, size=12))

    fig.update_layout(
        title="Comparable Sales vs Your Assessment",
        xaxis_title="Sale Date", yaxis_title="Adjusted Sale Price ($)",
        height=450, template="plotly_white",
        yaxis=dict(tickformat="$,.0f"),
    )
    return fig


def sales_histogram(sales_df: pd.DataFrame, user_tcv: float) -> go.Figure:
    """Histogram of sale prices with TCV and median lines."""
    if sales_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No comparable sales found", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        return fig

    prices = sales_df["Adj_Sale"].dropna()
    median = float(prices.median())

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=prices, nbinsx=15, marker_color=LIGHT_BLUE,
        marker_line=dict(width=1, color=NAVY),
        hovertemplate="Price range: $%{x:,.0f}<br>Count: %{y}<extra></extra>",
    ))

    fig.add_vline(x=user_tcv, line_dash="dash", line_color=RED, line_width=2,
                  annotation_text=f"Your TCV: ${user_tcv:,.0f}", annotation_position="top right",
                  annotation_font=dict(color=RED))
    fig.add_vline(x=median, line_dash="dot", line_color=GREEN, line_width=2,
                  annotation_text=f"Median: ${median:,.0f}", annotation_position="top left",
                  annotation_font=dict(color=GREEN))

    fig.update_layout(
        title="Sale Price Distribution",
        xaxis_title="Adjusted Sale Price ($)", yaxis_title="Count",
        xaxis=dict(tickformat="$,.0f"),
        height=400, template="plotly_white",
    )
    return fig


def land_trend_chart(land_trends: list[dict]) -> go.Figure:
    """Bar chart showing land values with adjustment factor annotations."""
    if not land_trends:
        fig = go.Figure()
        fig.add_annotation(text="No land value data", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        return fig

    years = [lt["year"] for lt in land_trends]
    current_vals = [lt.get("current_lv") for lt in land_trends]
    factors = [lt.get("adjust_factor") for lt in land_trends]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=years, y=current_vals, marker_color=NAVY,
        text=[f"${v:,.0f}" if v else "N/A" for v in current_vals],
        textposition="outside",
        hovertemplate="Year: %{x}<br>Land Value: $%{y:,.0f}<extra></extra>",
    ))

    # Add adjustment factor annotations
    for i, (y, f) in enumerate(zip(years, factors)):
        if f is not None:
            pct = (f - 1) * 100
            fig.add_annotation(
                x=y, y=(current_vals[i] or 0) * 0.5,
                text=f"Factor: {f:.4f}<br>({pct:+.1f}%)",
                showarrow=False, font=dict(size=11, color="white"),
            )

    fig.update_layout(
        title="Land Value Trend", xaxis_title="Year", yaxis_title="Land Value ($)",
        xaxis=dict(tickmode="array", tickvals=years),
        yaxis=dict(tickformat="$,.0f"),
        height=400, template="plotly_white",
    )
    return fig


def sales_coverage_chart(coverage: dict[int, int]) -> go.Figure:
    """Bar chart showing # of area sales in study per year."""
    years = sorted(coverage.keys())
    counts = [coverage[y] for y in years]
    colors = [RED if c == 0 else NAVY for c in counts]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=years, y=counts, marker_color=colors,
        text=[str(c) if c > 0 else "DROPPED" for c in counts],
        textposition="outside", textfont=dict(size=14),
        hovertemplate="Year: %{x}<br>Sales in study: %{y}<extra></extra>",
    ))

    fig.update_layout(
        title="Sales Study Coverage for Your Area",
        xaxis_title="Year", yaxis_title="# of Sales in Study",
        xaxis=dict(tickmode="array", tickvals=years),
        height=350, template="plotly_white",
    )
    return fig


def assessment_comparison_chart(user_tcv: float, ecf_adjusted: float | None,
                                median_sale: float | None) -> go.Figure:
    """Horizontal bar chart comparing TCV, ECF-adjusted, and median sale."""
    labels = []
    values = []
    colors = []

    if median_sale is not None and median_sale > 0:
        labels.append(f"Median Sale Price")
        values.append(median_sale)
        colors.append(GREEN)

    if ecf_adjusted is not None:
        labels.append("ECF-Adjusted Value")
        values.append(ecf_adjusted)
        colors.append(ORANGE)

    labels.append("Your Implied TCV")
    values.append(user_tcv)
    colors.append(RED)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=values, orientation="h", marker_color=colors,
        text=[f"${v:,.0f}" for v in values], textposition="outside",
        hovertemplate="%{y}: $%{x:,.0f}<extra></extra>",
    ))

    fig.update_layout(
        title="Value Comparison",
        xaxis_title="Value ($)", xaxis=dict(tickformat="$,.0f"),
        height=250, template="plotly_white", margin=dict(l=150),
    )
    return fig


def assessment_history_chart(history: list) -> go.Figure:
    """Line chart showing assessment history (Land, Building, Assessed, Taxable) over years."""
    if not history:
        fig = go.Figure()
        fig.add_annotation(text="No assessment history", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        return fig

    years = [h.year for h in history]

    traces = [
        ("Land Value", [h.land_value for h in history], GREEN),
        ("Building Value", [h.building_value for h in history], ORANGE),
        ("Assessed (SEV)", [h.assessed_value for h in history], NAVY),
        ("Taxable Value", [h.taxable_value for h in history], RED),
    ]

    fig = go.Figure()
    for name, values, color in traces:
        valid = [(y, v) for y, v in zip(years, values) if v is not None]
        if valid:
            fig.add_trace(go.Scatter(
                x=[p[0] for p in valid], y=[p[1] for p in valid],
                mode="lines+markers+text", name=name,
                line=dict(color=color, width=2), marker=dict(size=8),
                text=[f"${v:,.0f}" for _, v in valid],
                textposition="top center", textfont=dict(size=10),
                hovertemplate=f"{name}: $%{{y:,.0f}}<extra></extra>",
            ))

    fig.update_layout(
        title="Your Assessment History (from Record Card)",
        xaxis_title="Year", yaxis_title="Value ($)",
        xaxis=dict(tickmode="array", tickvals=years, dtick=1),
        yaxis=dict(tickformat="$,.0f"),
        height=450, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
