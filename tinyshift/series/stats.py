# Copyright (c) 2024-2025 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


from typing import Union, Tuple, List
import numpy as np
import scipy
import math


def hurst_exponent(
    X: Union[np.ndarray, List[float]],
) -> Tuple[float, float]:
    """
    Calculate the Hurst exponent using a rescaled range (R/S) analysis approach with p-value for random walk hypothesis.

    The Hurst exponent is a measure of long-term memory of time series. It relates
    to the autocorrelations of the time series and the rate at which these decrease
    as the lag between pairs of values increases.

    Parameters
    ----------
    X : Union[np.ndarray, List[float]]
        Input 1D time series data for which to calculate the Hurst exponent.
        Must contain at least 30 samples.

    Returns
    -------
    Tuple[float, float]
        (Hurst exponent, p-value for H=0.5 hypothesis)
        The estimated Hurst exponent value. Interpretation:
        - 0 < H < 0.5: Mean-reverting (anti-persistent) series
        - H = 0.5: Geometric Brownian motion (random walk)
        - 0.5 < H < 1: Trending (persistent) series with long-term memory
        - H = 1: Perfectly trending series
        p-value interpretation:
        - p < threshold: Reject random walk hypothesis (significant persistence/mean-reversion)
        - p >= threshold: Cannot reject random walk hypothesis

    Raises
    ------
    ValueError
        If input data has less than 30 samples (insufficient for reliable estimation).
    TypeError
        If input is not a list or numpy array.
    """
    X = np.asarray(X, dtype=np.float64)
    deltas = np.diff(X)
    size = len(deltas)

    if 30 > len(X):
        raise ValueError("Insufficient data points (minimum 30 required)")

    def _calculate_rescaled_ranges(
        deltas: np.ndarray, window_sizes: List[int]
    ) -> np.ndarray:
        """Helper function to calculate rescaled ranges (R/S) for each window size."""
        r_s = np.zeros(len(window_sizes), dtype=np.float64)

        for i, window_size in enumerate(window_sizes):
            n_windows = len(deltas) // window_size
            truncated_size = n_windows * window_size

            windows = deltas[:truncated_size].reshape(n_windows, window_size)

            means = np.mean(windows, axis=1, keepdims=True)
            std_devs = np.std(windows, axis=1, ddof=1)
            demeaned = windows - means
            cumulative_sums = np.cumsum(demeaned, axis=1)
            ranges = np.max(cumulative_sums, axis=1) - np.min(cumulative_sums, axis=1)

            r_s[i] = np.mean(ranges / std_devs)

        return r_s

    def _hypothesis_test_random_walk(hurst: float, se: float, n: int) -> float:
        """Helper function to test if Hurst exponent is significantly different from random_walk (0.5)"""
        random_walk = 0.5
        t_stat = (hurst - random_walk) / se
        ddof = n - 2
        return 2 * scipy.stats.t.sf(abs(t_stat), ddof)

    max_power = int(np.floor(math.log2(size)))
    window_sizes = [2**power for power in range(1, max_power + 1)]

    rescaled_ranges = _calculate_rescaled_ranges(deltas, window_sizes)

    log_sizes = np.log(window_sizes)
    log_r_s = np.log(rescaled_ranges)
    slope, _, _, _, se = scipy.stats.linregress(log_sizes, log_r_s)

    p_value = _hypothesis_test_random_walk(slope, se, len(window_sizes))

    return float(slope), float(p_value)


def relative_strength_index(
    X: Union[np.ndarray, List[float]],
    rolling_window: int = 14,
) -> np.ndarray:
    """
    Feature transformer that computes the Relative Strength Index (RSI) for a given time series.

    The RSI is a momentum oscillator that quantifies the magnitude and direction of recent movements in a time series.
    Its values range from 0 to 100 and are commonly used to indicate different momentum regimes.

    Parameters
    ----------
    X : array-like, shape (n_samples,)
        Time series data (e.g., closing prices).
    rolling_window : int, optional (default=14)
        The number of periods to use for calculating the RSI.

    Returns
    -------
    rsi : ndarray, shape (n_samples,)
        The RSI values for the time series.

    Notes
    -----
    - The RSI is calculated from the average gains and losses of returns over the specified rolling_window.
    - The first RSI value is computed after `rolling_window` periods.
    - Higher values indicate stronger positive momentum; lower values indicate stronger negative momentum.
    - Preserves the length of the input series; the first `rolling_window` values are initialized with the first computed RSI.
    """
    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 1:
        raise ValueError("Input data must be 1-dimensional")

    deltas = np.diff(X)
    seed = deltas[: rolling_window + 1]
    mean_gain = seed[seed >= 0].sum() / rolling_window
    mean_loss = -seed[seed < 0].sum() / rolling_window
    rs = mean_gain / mean_loss if mean_loss != 0.0 else 0.0
    rsi = np.zeros_like(X)
    rsi[:rolling_window] = 100.0 - 100.0 / (1.0 + rs)

    for i in range(rolling_window, len(X)):
        delta = deltas[i - 1]
        gain = max(delta, 0)
        loss = -min(delta, 0)
        mean_gain = (mean_gain * (rolling_window - 1) + gain) / rolling_window
        mean_loss = (mean_loss * (rolling_window - 1) + loss) / rolling_window
        rs = mean_gain / mean_loss if mean_loss != 0 else 0
        rsi[i] = 100.0 - 100.0 / (1.0 + rs)

    return rsi


def standardize_returns(X, log=True, standardize=True) -> np.ndarray:
    """
    Calculates and normalizes the returns of a time series.

    The function computes either logarithmic or simple returns from the
    input series and then standardizes the resulting return series
    (Z-score normalization).

    Parameters
    ----------
    X : array-like
        A 1-dimensional time series (e.g., prices, sales figures, volume).
    log : bool, default=True
        If True, calculates **logarithmic returns**: r_t = ln(X_t / X_{t-1}).
        If False, calculates **simple (percentage) returns**: R_t = (X_t / X_{t-1}) - 1.
    standardize : bool, default=True
        If True, standardizes the return series to have zero mean and unit variance.

    Returns
    -------
    norm : np.ndarray
        The normalized return series (with zero mean and unit standard deviation).

    Raises
    ------
    ValueError
        If the input data 'X' is not 1-dimensional.
    """
    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 1:
        raise ValueError("Input data must be 1-dimensional")

    ratios = X[1:] / X[:-1]
    returns = np.log(ratios) if log else ratios - 1
    return (returns - np.mean(returns)) / np.std(returns) if standardize else returns
