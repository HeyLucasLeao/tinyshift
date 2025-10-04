# Copyright (c) 2024-2025 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


from typing import Union, List, Tuple
import numpy as np
from scipy.signal import periodogram
from collections import Counter
import math


def foreca(
    X: Union[np.ndarray, List[float]],
) -> float:
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


def adi_cv(
    X: Union[np.ndarray, List[float]],
) -> Tuple[float, float]:
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


def sample_entropy(
    X: Union[np.ndarray, List[float]],
    m: int = 1,
    tolerance: float = None,
) -> np.ndarray:
    """
    Compute the Sample Entropy (SampEn) of a 1D time series.

    Sample Entropy is a measure of complexity or irregularity in a time series.
    It quantifies the likelihood that similar patterns in the data will not be followed by additional similar patterns.

    Parameters
    ----------
    X : array-like, shape (n_samples,)
        1D time series data.
    m : int
        Length of sequences to be compared (embedding dimension).
    tolerance : float, optional (default=None)
        Tolerance for accepting matches. If None, it is set to 0.1 * std(X).
    Returns
    -------
    sampen : float
        The Sample Entropy of the time series. Returns np.nan if A or B is zero.
    References
    ----------
    - Richman, J. S., & Moorman, J. R. (2000). Physiological time-series analysis using approximate entropy and sample entropy. American Journal of Physiology-Heart and Circulatory Physiology, 278(6), H2039-H2049.
    - Lake, D. E., Richman, J. S., Griffin, M. P., & Moorman, J. R. (2002). Sample entropy analysis of neonatal heart rate variability. American Journal of Physiology-Regulatory, Integrative and Comparative Physiology, 283(3), R789-R797.
    Notes
    -----
    - SampEn is less biased than Approximate Entropy (ApEn) and does not count self-matches.
    - Higher SampEn values indicate more complexity and irregularity in the time series.
    - The function assumes the input time series is 1-dimensional.
    - The function uses the Chebyshev distance (maximum norm) for comparing sequences.
    - If either A or B is zero, SampEn is undefined and np.nan is returned.
    """

    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 1:
        raise ValueError("Input data must be 1-dimensional")

    n = X.shape[0]

    if tolerance is None:
        tolerance = 0.2 * np.std(X)

    if m < 1:
        raise ValueError("m must be a positive integer")

    if tolerance <= 0:
        raise ValueError("tolerance must be a positive float")

    if m > n:
        raise ValueError("m must be less or equal to length of the time series")

    Xm = np.array([X[i : i + m] for i in range(n - m)])

    Xm1 = np.array([X[i : i + m + 1] for i in range(n - m - 1)])

    def count_matches(X_templates, tol):
        """
        Count the number of matching template pairs within the given tolerance. Chebyshev distance is used.

        Parameters
        ----------
        X_templates : ndarray, shape (N, m) or (N, m+1)
            Array of template vectors.
        tol : float
            Tolerance for accepting matches.
        Returns
        -------
        count : int
            Number of matching template pairs.
        """

        count = 0
        N = len(X_templates)
        for i in range(N):
            diff = np.abs(X_templates[i] - X_templates[i + 1 :])
            max_diff = np.max(diff, axis=1)
            count += np.sum(max_diff < tol)
        return count

    B = count_matches(Xm, tolerance)

    A = count_matches(Xm1, tolerance)

    if A > 0 and B > 0:
        sampen = -np.log(A / B)
    else:
        sampen = np.nan

    return sampen


def pattern_stability_index(
    X: Union[np.ndarray, List[float]],
    m: int = 1,
    tolerance=None,
) -> float:
    """
    Calculates the Pattern Stability Index based on Sample Entropy (SampEn).

    The Sample Entropy (hrate) measures the complexity and irregularity of a time series.
    This metric inverts the entropy to quantify the *regularity* or *predictability*
    of the series, showing the probability that a similar pattern will persist.

    The formula used is 1 / (2 ** hrate).

    Args:
        X (array-like): The time series data (e.g., standardized returns).
        m (int, optional): The embedding dimension (length of the pattern). Defaults to 1.
        tolerance (float, optional): The similarity criterion (r). If None, the
            implementation of sample_entropy will use its default (often 0.2 * std(X)).

    Returns:
        float: The Pattern Stability Index.
               - A value close to 1 indicates HIGH stability/regularity (highly predictable patterns).
               - A value close to 0 indicates LOW stability/regularity (highly random/complex series).
    """
    hrate = sample_entropy(X, m=m, tolerance=tolerance)
    return 1 / np.exp(hrate)


def permutation_entropy(
    X: Union[np.ndarray, List[float]],
    m: int = 3,
    delay: int = 1,
    normalize=True,
):
    """
    Calculate the Permutation Entropy of a time series.
    Parameters
    ----------
    X : array-like, shape (n_samples,)
        Time series data (e.g., closing prices).
        m : int, optional (default=3)
        The embedding dimension (length of the pattern).
        delay : int, optional (default=1)
        The time delay (spacing between elements in the pattern).
        normalize : bool, optional (default=False)
        If True, normalize the entropy to the range [0, 1].
        Returns
        -------
        float
            The Permutation Entropy of the time series.
        Notes
        -----
        - The Permutation Entropy quantifies the complexity of a time series based on the order relations between values.
        - It is calculated by mapping the time series to a sequence of ordinal patterns and computing the
        Shannon entropy of the distribution of these patterns.
        - Higher values indicate more complexity and randomness in the time series.
        - The function preserves the length of the input series.
    """
    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 1:
        raise ValueError("Input data must be 1-dimensional")
    if m < 2:
        raise ValueError("m must be at least 2")
    if delay < 1:
        raise ValueError("delay must be at least 1")
    if len(X) < (m - 1) * delay + 1:
        raise ValueError("Time series is too short for the given m and delay")

    N = X.shape[0] - delay * (m - 1)
    window_indices = [np.arange(i, i + delay * m, delay) for i in range(N)]
    X = np.argsort(X[window_indices], axis=1)
    patterns = Counter(map(tuple, X))
    probs = {k: v / sum(patterns.values()) for k, v in patterns.items()}
    probs = np.array(list(probs.values()))
    pe = -np.sum(probs * np.log2(probs))
    return pe / np.log2(math.factorial(m)) if normalize else pe


def maximum_achievable_predictability(
    X: Union[np.ndarray, List[float]],
    m: int = 3,
    delay: int = 1,
) -> float:
    """
    Calculates the Maximum Achievable Predictability (Πmax) based on the theoretical limit:
    Πmax = 1 - normalized Permutation Entropy (PE).
    Πmax ranges from 0 (completely unpredictable) to 1 (perfectly predictable).

    Parameters:
        X (array-like): The time series data (e.g., closing prices).
        m (int, optional): The embedding dimension (length of the pattern). Defaults to 3.
        delay (int, optional): The time delay (spacing between elements in the pattern). Defaults to 1.
    Returns:
        float: The Maximum Achievable Predictability (Πmax).

    - The function assumes the input time series is 1-dimensional.
    - Higher Πmax values indicate more predictable patterns in the time series.
    - The function uses the Permutation Entropy as a measure of complexity.
    """
    pe = permutation_entropy(X, m=m, delay=delay, normalize=True)

    return 1 - pe
