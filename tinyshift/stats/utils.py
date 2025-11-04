# Copyright (c) 2024-2025 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


import numpy as np
from typing import Union, List, Callable
import pandas as pd


def chebyshev_guaranteed_percentage(
    X: Union[np.ndarray, List[float]], interval: Union[np.ndarray, List[float]]
) -> float:
    """
    Computes the minimum percentage of data within a given interval using Chebyshev's inequality.

    Chebyshev's theorem guarantees that for any distribution, at least (1 - 1/k²) of the data lies
    within 'k' standard deviations from the mean. The coefficient 'k' is computed for each bound
    (lower and upper) independently, and the conservative (smaller) value is chosen to ensure a
    valid lower bound.

    Parameters:
    ----------
    X : array-like
        Input numerical data.
    interval : tuple (lower, upper)
        The interval of interest (lower and upper bounds). Use None for unbounded sides.

    Returns:
    -------
    float
        The minimum fraction (between 0 and 1) of data within the interval.
        Returns 0 if the interval is too wide (k ≤ 1), where the theorem provides no meaningful bound.

    Notes:
    -----
    - If `lower` is None, the interval is unbounded on the left.
    - If `upper` is None, the interval is unbounded on the right.
    """

    X = np.asarray(X)
    mu = np.mean(X)
    std = np.std(X)
    lower, upper = interval
    k_values = []
    if lower is not None:
        k_lower = (mu - lower) / std
        k_values.append(k_lower)
    if upper is not None:
        k_upper = (upper - mu) / std
        k_values.append(k_upper)
    k = float(min(k_values))
    return 1 - (1 / (k**2)) if k > 1 else 0


def rolling_window(
    X: Union[np.ndarray, List[float]],
    window_size: int = 60,
    func: Callable = None,
    **kwargs,
) -> np.ndarray:
    """
    Apply a function over a trailing (rolling) window of a 1D time series.

    Parameters
    ----------
    X : array-like, shape (n_samples,)
        1D time series data (e.g., log-prices).
    window_size : int, optional (default=60)
        Size of the rolling window (must be >= 3).
    func : Callable
        Function to apply to each window. Must accept a 1D array as first argument.
    **kwargs
        Additional keyword arguments to pass to `func`.

    Returns
    -------
    result : ndarray, shape (n_samples - window_size + 1,)
        Array of function values for each rolling window.
    """
    if window_size < 2:
        raise ValueError("window_size must be >= 2")

    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 1:
        raise ValueError("Input data must be 1-dimensional")

    window_indices = [
        np.arange(i, i + window_size) for i in range(X.shape[0] - window_size + 1)
    ]

    windows = X[window_indices]

    result = np.array([func(window, **kwargs) for window in windows])

    return np.concatenate(([result[0]] * (window_size - 1), result))


def expanding_window(
    X: Union[np.ndarray, List[float]],
    func: Callable = None,
    window_size: int = 1,
    **kwargs,
) -> np.ndarray:
    """
    Apply a function over an expanding window of a 1D time series.

    Parameters
    ----------
    X : array-like, shape (n_samples,)
        1D time series data (e.g., log-prices).
    func : Callable
        Function to apply to each window. Must accept a 1D array as first argument.
    window_size : int, optional (default=1)
        Minimum window size to start the expansion. The first window will contain
        `window_size` elements, and subsequent windows will expand by one element each.
    **kwargs
        Additional keyword arguments to pass to `func`.

    Returns
    -------
    result : ndarray, shape (n_samples - window_size + 1,)
        Array of function values for each expanding window, starting from `window_size`.
    """
    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 1:
        raise ValueError("Input data must be 1-dimensional")

    if window_size < 1:
        raise ValueError("window_size must be >= 1")

    if window_size > len(X):
        raise ValueError("window_size cannot be larger than the length of X")

    result = np.array(
        [
            func(X[: window_size + i], **kwargs)
            for i in range(X.shape[0] - window_size + 1)
        ]
    )

    return np.concatenate(([result[0]] * (window_size - 1), result))


def jacknife(
    X: Union[np.ndarray, List[float]],
    func: Callable = None,
    **kwargs,
) -> np.ndarray:
    """
    Apply a function using the jackknife approach on a 1D time series.
    Parameters
    ----------
    X : array-like, shape (n_samples,)
        1D time series data (e.g., log-prices).
    func : Callable
        Function to apply to each jackknife sample. Must accept a 1D array as first argument.
    **kwargs
        Additional keyword arguments to pass to `func`.
    Returns
    -------
    result : ndarray, shape (n_samples,)
        Array of function values for each jackknife sample.
    """
    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 1:
        raise ValueError("Input data must be 1-dimensional")

    result = np.array([func(np.delete(X, i), **kwargs) for i in range(X.shape[0])])

    return result


def mad(x):
    """
    Calculate the Median Absolute Deviation (MAD) of a 1D array.

    The MAD is a robust measure of variability that is less sensitive to outliers
    than the standard deviation. It is defined as the median of the absolute
    deviations from the data's median.

    Parameters
    ----------
    x : array-like
        Input data array.

    Returns
    -------
    float
        The median absolute deviation of the input data.
    """
    return np.median(np.absolute(x - np.median(x)))


def generate_lag(
    X: Union[np.ndarray, List[float]],
    lag=1,
):
    """
    Generate lagged differences for a 1D time series.

    Parameters
    ----------
    X : array-like, shape (n_samples,)
        1D time series data.
    lag : int, optional (default=1)
        The lag interval.

    Returns
    -------
    result : ndarray, shape (n_samples,)
        Array containing NaNs for the first 'lag' elements followed by the lagged differences.
    """
    X = np.asarray(X, dtype=np.float64)

    if X.ndim > 1:
        raise ValueError("Input array must be one-dimensional.")

    return np.concatenate((np.nan * np.ones(lag), (X[lag:] - X[:-lag])))


def remove_leading_zeros(group):
    """
    Removes leading zeros from a time series group.

    Parameters
    ----------
    group : pandas.DataFrame
        DataFrame containing time series data with a 'y' column.

    Returns
    -------
    pandas.DataFrame
        DataFrame with leading zeros removed, starting from the first non-zero value.
    """
    first_non_zero_index = group["y"].ne(0).idxmax()
    return group.loc[first_non_zero_index:]


def is_obsolete(group, days_obsoletes):
    """
    Determine if a time series group is obsolete based on recent data.

    Parameters
    ----------
    group : pandas.DataFrame
        DataFrame containing time series data with 'ds' (date) and 'y' (value) columns.
    days_obsoletes : int
        Number of days to look back from the last date to check for obsolescence.

    Returns
    -------
    bool
        True if all values in the recent period (last 'days_obsoletes' days) are zero,
        False otherwise.
    """
    last_date = group["ds"].max()
    cutoff_date = last_date - pd.Timedelta(days=days_obsoletes)
    recent_data = group[group["ds"] >= cutoff_date]
    return (recent_data["y"] == 0).all()
