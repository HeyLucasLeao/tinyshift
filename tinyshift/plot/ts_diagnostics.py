# Copyright (c) 2024-2025 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa import seasonal
from statsmodels.tsa.stattools import adfuller, acf, pacf
from tinyshift.series import trend_significance
import plotly.subplots as sp
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
from typing import Union, List, Optional
import pandas as pd


def seasonal_decompose(
    X: Union[np.ndarray, List[float], pd.Series],
    model: str = "additive",
    filt: Optional[np.ndarray] = None,
    period: int = None,
    two_sided: bool = True,
    extrapolate_trend: int = 0,
    height: int = 1200,
    width: int = 1300,
    ljung_lags: int = 10,
    fig_type: Optional[str] = None,
):
    """
    Performs seasonal decomposition of a time series and plots the components.

    This function uses the `seasonal_decompose` method from statsmodels to separate the
    time series into observed, trend, seasonal, and residual components.
    Additionally, it calculates trend significance and the Ljung-Box test for
    residuals, displaying a summary in the plot.

    Parameters
    ----------
    X : array-like
        The time series to be decomposed. Expected to be an object with an index
        (e.g., pandas Series).
    model : {"additive", "multiplicative"}, default="additive"
        Type of seasonal model. If "additive", $X = T + S + R$. If
        "multiplicative", $X = T \cdot S \cdot R$.
    filt : array-like, optional
        Moving average filter for calculating the trend component. By default,
        a symmetric filter is used.
    period : int, optional
        Period of the series (number of observations per cycle). If `None` and `X` is
        a pandas Series, the period is inferred from the index frequency.
    two_sided : bool, default=True
        If `True` (default), uses a centered moving average filter. If `False`, uses
        a causal filter (future only).
    extrapolate_trend : int or str, default=0
        Number of points at the beginning and end to extrapolate the trend. If 0 (default),
        the trend is `NaN` at these extremes.
    height : int, default=1200
        Figure height in pixels.
    width : int, default=1300
        Figure width in pixels.
    ljung_lags : int, default=10
        The number of lags to be used in the Ljung-Box test for residuals.
    fig_type : str, optional
        Plotly figure output type. Passed to `fig.show()`.
        E.g.: 'json', 'html', 'notebook'.

    Returns
    -------
    plotly.graph_objects.Figure or None
        Returns the Plotly Figure object if `fig_type` is `None` or the result
        of the `fig.show(fig_type)` call.

    Notes
    -----
    The resulting plot is a Plotly `make_subplots` with 5 subplots:
    - Observed
    - Trend
    - Seasonal
    - Residuals
    - Summary (includes trend significance - $R^2$ and p-value - and the
      Ljung-Box test for residual autocorrelation).
    """

    index = X.index if hasattr(X, "index") else list(range(len(X)))

    if X.ndim != 1:
        raise ValueError("Input data must be 1-dimensional")

    colors = px.colors.qualitative.T10
    num_colors = len(colors)

    result = seasonal.seasonal_decompose(
        X,
        model=model,
        filt=filt,
        period=period,
        two_sided=two_sided,
        extrapolate_trend=extrapolate_trend,
    )
    fig = sp.make_subplots(
        rows=5,
        cols=1,
        subplot_titles=[
            "Observed",
            "Trend",
            "Seasonal",
            "Residuals",
            "Summary",
        ],
        row_heights=[4, 4, 4, 4, 1],
    )

    r_squared, p_value = trend_significance(X)
    trend_results = f"R²={r_squared:.4f}, p={p_value:.4f}"
    resid = result.resid[~np.isnan(result.resid)]
    ljung_box = acorr_ljungbox(resid, lags=[ljung_lags])
    ljung_stat, p_value = (
        ljung_box["lb_stat"].values[0],
        ljung_box["lb_pvalue"].values[0],
    )
    ljung_box = f"Stats={ljung_stat:.4f}, p={p_value:.4f}"
    summary = "<br>".join(
        [
            f"<b>{k}</b>: {v}"
            for k, v in {
                "Trend Significance": trend_results,
                "Ljung-Box Test": ljung_box,
            }.items()
        ]
    )

    for i, col in enumerate(["observed", "trend", "seasonal", "resid"]):
        color = colors[(i - 1) % num_colors]
        fig.add_trace(
            go.Scatter(
                x=index,
                y=getattr(result, col),
                mode="lines",
                hovertemplate=f"{col.capitalize()}: " + "%{y}<extra></extra>",
                line=dict(color=color),
            ),
            row=i + 1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(x=[0], y=[0], text=[summary], mode="text", showlegend=False),
        row=5,
        col=1,
    )

    fig.update_xaxes(visible=False, row=5, col=1)
    fig.update_yaxes(visible=False, row=5, col=1)

    fig.update_layout(
        title="Seasonal Decomposition",
        height=height,
        width=width,
        showlegend=False,
        hovermode="x",
    )

    return fig.show(fig_type)


def stationarity_check(
    df: Union[pd.DataFrame, pd.Series],
    height: int = 1200,
    width: int = 1300,
    nlags: int = 30,
    fig_type: Optional[str] = None,
):
    """
    Creates interactive ACF and PACF plots with ADF test results for multiple series.

    This function generates a comprehensive diagnostic visualization to assess the
    stationarity and autocorrelation structure of multiple time series in a single panel.
    The plot includes the series itself, its autocorrelation function (ACF) and partial
    autocorrelation function (PACF), and a summary of the Augmented Dickey-Fuller (ADF)
    test results.

    Parameters
    ----------
    df : pandas.DataFrame, pandas.Series, or list
        Input data containing the time series. Can be:
        - DataFrame: Multiple columns will be analyzed
        - Series: Will be converted to single-column DataFrame
    height : int, default=1200
        Figure height in pixels.
    width : int, default=1300
        Figure width in pixels.
    nlags : int, default=30
        Number of lags to include in ACF and PACF calculations.
    fig_type : str, optional
        Plotly figure output type. Passed to `fig.show()`.
        E.g.: 'json', 'html', 'notebook'.

    Returns
    -------
    plotly.graph_objects.Figure or None
        Returns the Plotly Figure object if `fig_type` is `None` or the result
        of the `fig.show(fig_type)` call.

    Notes
    -----
    The function generates a subplot structure where each row corresponds to a
    variable and displays:
    1. The Time Series.
    2. The Autocorrelation Function (ACF).
    3. The Partial Autocorrelation Function (PACF).
    The last row contains a summary of the ADF test results (statistic and p-value)
    for each variable, used to check for stationarity.

    Confidence bands are shown on ACF and PACF plots at ±1.96/√N level.
    """

    if isinstance(df, pd.Series):
        series_name = df.name if df.name is not None else "Value"
        df = df.to_frame(name=series_name)

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Input must be a pandas Series, pandas DataFrame, or a list (of lists)."
        )

    N = len(df.columns)
    colors = px.colors.qualitative.T10
    num_colors = len(colors)

    def create_acf_pacf_traces(X, nlags=30, color=None):
        """
        Helper function to create ACF and PACF traces with confidence intervals.
        """

        N = len(X)
        conf = 1.96 / np.sqrt(N)
        acf_vals = acf(X, nlags=nlags)
        pacf_vals = pacf(X, nlags=nlags, method="yw")

        acf_bar = go.Bar(x=list(range(len(acf_vals))), y=acf_vals, marker_color=color)
        pacf_bar = go.Bar(
            x=list(range(len(pacf_vals))), y=pacf_vals, marker_color=color
        )

        band_upper = go.Scatter(
            x=list(range(nlags + 1)),
            y=[conf] * (nlags + 1),
            mode="lines",
            line=dict(color="gray", dash="dash"),
            showlegend=False,
        )
        band_lower = go.Scatter(
            x=list(range(nlags + 1)),
            y=[-conf] * (nlags + 1),
            mode="lines",
            line=dict(color="gray", dash="dash"),
            showlegend=False,
        )

        return acf_bar, pacf_bar, band_upper, band_lower

    subplot_titles = []
    for var in df.columns:
        subplot_titles.extend([f"Series ({var})", f"ACF ({var})", f"PACF ({var})"])
    subplot_titles.extend(["ADF Results Summary", "", ""])

    fig = sp.make_subplots(rows=N + 1, cols=3, subplot_titles=subplot_titles)

    adf_results = {}

    for i, var in enumerate(df.columns, start=1):
        X = df[var].dropna()
        adf_stat, p_value = adfuller(X)[:2]
        adf_results[var] = f"ADF={adf_stat:.4f}, p={p_value:.4f}"
        color = colors[(i - 1) % num_colors]

        fig.add_trace(
            go.Scatter(
                x=X.index,
                y=X,
                mode="lines",
                name=var,
                showlegend=False,
                line=dict(color=color),
            ),
            row=i,
            col=1,
        )

        acf_values, pacf_values, conf_up, conf_lo = create_acf_pacf_traces(
            X,
            color=color,
            nlags=nlags,
        )

        fig.add_trace(acf_values, row=i, col=2)
        fig.add_trace(pacf_values, row=i, col=3)
        fig.add_trace(conf_up, row=i, col=2)
        fig.add_trace(conf_lo, row=i, col=2)
        fig.add_trace(conf_up, row=i, col=3)
        fig.add_trace(conf_lo, row=i, col=3)

    adf_text = "<br>".join([f"<b>{k}</b>: {v}" for k, v in adf_results.items()])

    fig.add_trace(
        go.Scatter(x=[0], y=[0], text=[adf_text], mode="text", showlegend=False),
        row=N + 1,
        col=1,
    )

    fig.update_layout(
        title="ACF/PACF with ADF Summary",
        height=height,
        width=width,
        showlegend=False,
    )

    for row in range(1, N + 1):
        fig.update_xaxes(title_text="Date", row=row, col=1)
        fig.update_yaxes(title_text="Value", row=row, col=1)
        fig.update_xaxes(title_text="Lag", row=row, col=2)
        fig.update_xaxes(title_text="Lag", row=row, col=3)
        fig.update_yaxes(title_text="ACF", row=row, col=2)
        fig.update_yaxes(title_text="PACF", row=row, col=3)

    fig.update_xaxes(visible=False, row=N + 1, col=1)
    fig.update_yaxes(visible=False, row=N + 1, col=1)

    return fig.show(fig_type)
