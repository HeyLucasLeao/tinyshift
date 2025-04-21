import numpy as np


def chebyshev_guaranteed_percentage(data, interval):
    """
    Computes the minimum percentage of data within a given interval using Chebyshev's inequality.

    Chebyshev's theorem guarantees that for any distribution, at least (1 - 1/k²) of the data lies
    within 'k' standard deviations from the mean. The coefficient 'k' is computed symmetrically
    for both bounds, and the conservative (smaller) value is chosen to ensure a valid lower bound.

    Parameters:
    -----------
    data : array-like
        Input numerical data.
    interval : tuple (lower, upper)
        The interval of interest (lower and upper bounds).

    Returns:
    --------
    float
        The minimum fraction (between 0 and 1) of data within the interval.
        Returns 0 if the interval is too wide (k ≤ 1), where the theorem provides no meaningful bound.

    """
    data = np.asarray(data)
    mu = np.mean(data)
    std = np.std(data)
    lower, upper = interval
    k_lower = (mu - lower) / std
    k_upper = (upper - mu) / std
    k = min(k_lower, k_upper)
    return 1 - (1 / (k**2)) if k > 1 else 0
