# Copyright (c) 2024-2025 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


import plotly.graph_objects as go
import plotly.express as px
from typing import Optional
import numpy as np
from sklearn.utils.validation import check_array
import pandas as pd


def corr_heatmap(
    X: np.ndarray, width: int = 1600, height: int = 1600, fig_type: Optional[str] = None
) -> go.Figure:
    """
    Generate an interactive correlation heatmap for a DataFrame using Plotly Express.

    This function visualizes the correlation matrix between all numeric columns in the DataFrame,
    with values displayed on each cell and a diverging color scale (blue for negative, red for positive correlations).

    Parameters:
    -----------
    df : np.ndarray
        A matrix containing numeric columns to compute correlations.
        Non-numeric columns will be automatically excluded.
    width : int, optional
        Width of the figure in pixels (default: 1600)
    height : int, optional
        Height of the figure in pixels (default: 1600)
    fig_type : str, optional
        Display type for the figure (particularly useful in Jupyter notebooks).
        Common options: None (default), 'notebook', or other Plotly-supported renderers.
        (default: None)

    Returns:
    --------
    None
        Displays the heatmap directly.

    Examples:
    --------
    >>> # Basic usage with default size
    >>> corr_heatmap(df)

    >>> # Custom size
    >>> corr_heatmap(df, width=1200, height=1200)

    >>> # For Jupyter notebook display
    >>> corr_heatmap(df, fig_type='notebook')

    Notes:
    ------
    - The correlation matrix is computed using pandas.DataFrame.corr() (Pearson correlation)
    - The color scale ranges from -1 (perfect negative correlation) to +1 (perfect positive correlation)
    - Diagonal elements will always be 1 (perfect self-correlation)
    - Only numeric columns are included in the calculation (equivalent to numeric_only=True)
    """
    feature_names_in_ = getattr(X, "columns", None)
    X = check_array(X, ensure_2d=True, dtype=np.float64, copy=True)

    if feature_names_in_ is None:
        feature_names_in_ = [f"feature_{i}" for i in range(X.shape[1])]

    corr = np.corrcoef(X, rowvar=False)

    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="equal",
        color_continuous_scale="RdBu_r",
        title="Heatmap of Correlation Coefficients",
        labels=dict(color="Correlation"),
        x=feature_names_in_,
        y=feature_names_in_,
    )

    fig.update_layout(
        width=width,
        height=height,
        xaxis_title="Features",
        yaxis_title="Features",
        title_x=0.5,
    )

    return fig.show(fig_type)


def corr_barplot(
    X: np.ndarray,
    y: np.ndarray,
    target_name: str = "target",
    width: int = 800,
    height: int = 600,
    color_scale: str = "RdBu",
    top_n: int = None,
    threshold: float = None,
    fig_type: str = None,
) -> go.Figure:
    """
    Plot correlation coefficients between features and target using Plotly.

    Parameters:
    -----------
    X : pd.DataFrame or np.ndarray
        Feature matrix. If DataFrame, column names will be used as feature names.
        If ndarray, features will be named automatically as 'feature_0', 'feature_1', etc.
    y : pd.Series or np.ndarray
        Target variable (1D array-like)
    target_name : str, optional
        Name to display for target variable (default: 'target')
    width : int, optional
        Figure width in pixels (default: 800)
    height : int, optional
        Figure height in pixels (default: 600)
    color_scale : str, optional
        Color scale for the plot (default: 'RdBu')
        Can be any valid Plotly continuous color scale name
    top_n : int, optional
        If specified, plot only top N features with highest absolute correlation
        (default: None shows all features)
    threshold : float, optional
        If specified, plot only features with absolute correlation >= threshold
        (default: None shows all features)
    fig_type : str, optional
        Figure output type (for Jupyter notebooks, can be 'notebook' or None)
        (default: None shows the figure in a new window)

    Returns:
    --------
    None
        Displays the plot directly

    Examples:
    --------
    >>> # With DataFrame/Series
    >>> barplot_corr_with_target(X_df, y_series)

    >>> # With numpy arrays
    >>> barplot_corr_with_target(X_array, y_array, target_name='price')

    >>> # Showing only top 10 features
    >>> barplot_corr_with_target(X, y, top_n=10)

    >>> # Only features with correlation >= 0.3
    >>> barplot_corr_with_target(X, y, threshold=0.3)
    """

    feature_names_in_ = getattr(X, "columns", None)
    target_name = getattr(y, "name", target_name)
    X = check_array(X, ensure_2d=True, dtype=np.float64, copy=True)

    if feature_names_in_ is None:
        feature_names_in_ = [f"feature_{i}" for i in range(X.shape[1])]

    data = np.column_stack((X, y))
    corr = np.corrcoef(data, rowvar=False)
    corr_target = corr[:-1, -1]
    corr_series = pd.Series(corr_target, index=feature_names_in_, name="correlation")
    corr_sorted = corr_series.sort_values(key=np.abs, ascending=False)

    if top_n is not None:
        corr_sorted = corr_sorted.head(top_n)
    if threshold is not None:
        corr_sorted = corr_sorted[np.abs(corr_sorted) >= threshold]

    fig = px.bar(
        corr_sorted,
        orientation="h",
        color=corr_sorted,
        color_continuous_scale=color_scale,
        title=f"Correlation Coefficients with {target_name}",
        labels={"x": "Correlation Coefficient", "y": "Feature"},
        width=width,
        height=height,
        text=np.round(corr_sorted, 3),
    )

    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        coloraxis_colorbar={
            "title": "Correlation",
            "tickvals": [-1, -0.5, 0, 0.5, 1],
            "ticktext": ["-1 (Neg)", "-0.5", "0", "0.5", "1 (Pos)"],
        },
        hovermode="y",
        uniformtext_minsize=8,
        uniformtext_mode="hide",
    )

    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="grey")

    return fig.show(fig_type)
