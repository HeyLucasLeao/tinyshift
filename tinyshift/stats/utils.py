# Copyright (c) 2024-2025 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


import numpy as np
from typing import Union, List


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
