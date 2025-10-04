# Copyright (c) 2024-2025 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


from typing import Union, List
import numpy as np
from tinyshift.stats import StatisticalInterval
from tinyshift.stats import trailing_window


def hampel_filter(
    X: Union[np.ndarray, List[float]],
    rolling_window: int = 3,
    factor: float = 3.0,
    scale: float = 1.4826,
) -> np.ndarray:
    """
    Identify outliers using a vectorized implementation of the Hampel filter.

    The Hampel filter is a robust outlier detection method that uses the median and
    median absolute deviation (MAD) of a rolling window to identify points that
    deviate significantly from the local trend. This version uses vectorized operations
    for improved performance.

    Parameters
    ----------
    X : ndarray of shape (n_samples,) or list of float
        Input 1D data to be filtered.
    rolling_window : int, default=3
        Size of the rolling window (must be odd and >= 3).
    factor : float, default=3.0
        Recommended values for common distributions (95% confidence):
        - Normal distribution: 3.0 (default)
        - Laplace distribution: 2.3
        - Cauchy distribution: 3.4
        - Exponential distribution: 3.6
        - Uniform distribution: 3.9
        Number of scaled MADs from the median to consider as outlier.
    scale : float, default=1.4826
        Scaling factor for MAD to make it consistent with standard deviation.
        Recommended values for different distributions:
        - Normal distribution: 1.4826 (default)
        - Uniform distribution: 1.16
        - Laplace distribution: 2.04
        - Exponential distribution: 2.08
        - Cauchy distribution: 1.0 (MAD is already consistent)
        - These values make the MAD scale estimator consistent with the standard
        deviation for the respective distribution.

    Returns
    -------
    outliers : ndarray of shape (n_samples,)
        Boolean array indicating outliers (True) and inliers (False).

    Raises
    ------
    ValueError
        If rolling_window is even or too small.
        If input data is not 1-dimensional.

    Notes
    -----
    The scale factor is chosen such that for large samples from the specified
    distribution, the median absolute deviation (MAD) multiplied by the scale
    factor approaches the standard deviation of the distribution.
    This implementation uses vectorized operations for better performance
    compared to the iterative version.
    """

    if rolling_window < 3:
        raise ValueError("rolling_window must be >= 3")
    if rolling_window % 2 == 0:
        raise ValueError("rolling_window must be odd")

    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 1:
        raise ValueError("Input data must be 1-dimensional")

    is_outlier = np.zeros(X.shape[0], dtype=bool)
    half_window = rolling_window // 2
    center_indices = range(half_window, X.shape[0] - half_window)

    window_indices = [
        np.arange(i - half_window, i + half_window + 1) for i in center_indices
    ]
    windows = X[window_indices]

    medians = np.median(windows, axis=1)
    mads = np.median(np.abs(windows - medians[:, None]), axis=1)
    thresholds = factor * mads * scale

    for i, idx in enumerate(center_indices):
        if abs(X[idx] - medians[i]) > thresholds[i]:
            is_outlier[idx] = True

    return is_outlier


def bollinger_bands(
    X: Union[np.ndarray, List[float]],
    rolling_window: int = 20,
    center: int = np.mean,
    spread: int = np.std,
    factor: int = 2,
) -> np.ndarray:
    """
    Feature transformer that computes the Bollinger Bands for a given time series.
    Bollinger Bands consist of a middle band (simple moving average) and two outer bands
    that are a specified number of standard deviations away from the middle band.
    The bands help identify periods of high and low volatility in the time series.

    Parameters
    ----------
    X : array-like, shape (n_samples,)
        Time series data (e.g., closing prices).
    rolling_window : int, optional (default=20)
        The number of periods to use for calculating the moving average and standard deviation.
    factor : float, optional (default=2)
        The number of standard deviations to use for the upper and lower bands.
    center : callable, optional (default=np.mean)
        Function to compute the center (e.g., mean or median) of the window.
    spread : callable, optional (default=np.std)
        Function to compute the spread (e.g., standard deviation or MAD) of the window.

    Returns
    -------
    outliers : ndarray, shape (n_samples,)
        Boolean array indicating outliers (True) and inliers (False).

    Notes
    -----
    - The Bollinger Bands are calculated using a rolling window approach.
    - Outliers are points outside the upper or lower band.
    """

    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 1:
        raise ValueError("Input data must be 1-dimensional")

    is_outlier = np.zeros(X.shape[0], dtype=bool)
    bounds = trailing_window(
        X,
        rolling_window=rolling_window,
        func=StatisticalInterval.calculate_interval,
        center=center,
        spread=spread,
        factor=factor,
    )

    is_outlier = np.where((X < bounds[:, 0]) | (X > bounds[:, 1]), True, is_outlier)

    return is_outlier
