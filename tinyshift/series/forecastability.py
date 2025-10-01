# Copyright (c) 2024-2025 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


from typing import Union, List
import numpy as np
from scipy.signal import periodogram


def foreca(X: Union[np.ndarray, List[float]]) -> float:
    """
    Calculate the Forecastable Component Analysis (ForeCA) omega index for a given signal.

    The omega index (ω) measures how forecastable a time series is, ranging from 0
    (completely noisy/unforecastable) to 1 (perfectly forecastable). It is based on
    the spectral entropy of the signal's power spectral density (PSD).

    Parameters
    ----------
    X : Union[np.ndarray, List[float]]
        Input 1D time series data for which to calculate the forecastability measure.
        The signal should be stationary for meaningful results.

    Returns
    -------
    float
        The omega forecastability index (ω), where:
        - ω ≈ 0: Signal is noise-like and not forecastable
        - ω ≈ 1: Signal has strong periodic components and is highly forecastable

    Notes
    -----
    The calculation involves:
    1. Computing the power spectral density (PSD) via periodogram
    2. Normalizing the PSD to sum to 1 (creating a probability distribution)
    3. Calculating the spectral entropy of this distribution
    4. Normalizing against maximum possible entropy
    5. Subtracting from 1 to get forecastability measure

    References
    ----------
    [1] Goerg (2013), "Forecastable Component Analysis" (JMLR)
    [2] Hyndman et al. (2015), "Large unusual observations in time series"
    [3] Manokhin (2025), "Mastering Modern Time Series Forecasting: The Complete Guide to
        Statistical, Machine Learning & Deep Learning Models in Python", Ch. 2.4.12
    """
    _, psd = periodogram(X)
    psd = psd / np.sum(psd)
    entropy = -np.sum(psd * np.log2(psd + 1e-12))
    max_entropy = np.log2(len(psd))
    omega = 1 - (entropy / max_entropy)
    return float(omega)


def adi_cv(X):
    """
    Computes two key metrics for analyzing time series data: Average Demand Interval (ADI)
    and Coefficient of Variation (CV).

    1. Average Demand Interval (ADI): Indicates the average number of periods between nonzero values in a time series.
       - Higher ADI suggests more periods of zero or low values, indicating potential sparsity or infrequent activity.
       - ADI = n / n_nonzero, where n is the total number of periods and n_nonzero is the count of nonzero values.

    2. Coefficient of Variation (CV): The squared ratio of the standard deviation to the mean of the time series.
       - Provides a normalized measure of dispersion, allowing for comparison across different time series regardless of their scale.
       - Higher CV indicates greater variability relative to the mean.
       - CV = (std(X) / mean(X)) ** 2

    Parameters
    ----------
    X : array-like, shape (n_samples,)
        Time series data (e.g., demand, sales, or other metrics).

    Returns
    -------
    adi : float
        Average Demand Interval for the time series.
    cv : float
        Squared Coefficient of Variation for the time series.

    Notes
    -----
    - ADI thresholds:
        * Low ADI < 1.32 (frequent activity)
        * High ADI >= 1.32 (infrequent activity)
    - CV thresholds:
        * Low CV < 0.49 (low variability)
        * High CV >= 0.49 (high variability)
    - Classification of time series:
        * "Smooth":      Low ADI, Low CV — consistent activity, low variability, highly predictable.
        * "Intermittent":High ADI, Low CV — infrequent but regular activity, forecastable with specialized methods (e.g., Croston's, ADIDA, IMAPA).
        * "Erratic":     Low ADI, High CV — regular activity but high variability, high uncertainty.
        * "Lumpy":       High ADI, High CV — periods of inactivity followed by bursts, challenging to forecast.
    """
    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 1:
        raise ValueError("Input data must be 1-dimensional")

    n = X.shape[0]
    n_nonzero = np.count_nonzero(X)
    adi = n / n_nonzero
    cv = (np.std(X) / np.mean(X)) ** 2

    return adi, cv
