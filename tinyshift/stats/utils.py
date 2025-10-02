# Copyright (c) 2024-2025 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


import numpy as np
from typing import Union, List, Callable


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


def trailing_window(
    X: Union[np.ndarray, List[float]],
    rolling_window: int = 60,
    func: Callable = None,
    **kwargs,
) -> np.ndarray:
    """
    Apply a function over a trailing (rolling) window of a 1D time series.

    Parameters
    ----------
    X : array-like, shape (n_samples,)
        1D time series data (e.g., log-prices).
    rolling_window : int, optional (default=60)
        Size of the rolling window (must be >= 3).
    func : Callable
        Function to apply to each window. Must accept a 1D array as first argument.
    **kwargs
        Additional keyword arguments to pass to `func`.

    Returns
    -------
    result : ndarray, shape (n_samples - rolling_window + 1,)
        Array of function values for each rolling window.
    """
    if rolling_window < 2:
        raise ValueError("rolling_window must be >= 2")

    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 1:
        raise ValueError("Input data must be 1-dimensional")

    window_indices = [
        np.arange(i, i + rolling_window) for i in range(X.shape[0] - rolling_window + 1)
    ]

    windows = X[window_indices]

    result = np.array([func(window, **kwargs) for window in windows])

    return np.concatenate(([result[0]] * (rolling_window - 1), result))
