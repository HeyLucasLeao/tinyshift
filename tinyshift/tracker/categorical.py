# Copyright (c) 2024-2025 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from .base import BaseModel
from typing import Callable, Tuple, Union, List
from collections import Counter


def chebyshev(a, b):
    """
    Compute the Chebyshev distance between two distributions.
    """
    return np.max(np.abs(a - b))


def psi(observed, expected, epsilon=1e-4):
    """
    Calculate Population Stability Index (PSI) between two distributions.
    """
    observed = np.clip(observed, epsilon, 1)
    expected = np.clip(expected, epsilon, 1)
    return np.sum((observed - expected) * np.log(observed / expected))


class CatDrift(BaseModel):
    def __init__(
        self,
        X: Union[pd.Series, List[np.ndarray], List[list]],
        func: str = "chebyshev",
        drift_limit: Union[str, Tuple[float, float]] = "stddev",
        method: str = "expanding",
        window_size: int = None,
    ):
        """
        A tracker for identifying drift in categorical data over time.

        The tracker uses a reference dataset to compute a baseline distribution and compares
        subsequent data for deviations based on a distance metric and drift limits.

        Available distance metrics:
        - 'chebyshev': Maximum absolute difference between category probabilities
        - 'jensenshannon': Jensen-Shannon divergence (symmetric, sqrt of JS distance)
        - 'psi': Population Stability Index (sensitive to small probability changes)

        Parameters
        ----------
        X : Union[pd.Series, List[np.ndarray], List[list]]
            Input categorical data. For time series, each element represents a period's
            categorical observations.
        func : str, default='chebyshev'
            Distance metric to use for drift detection.
        statistic : Callable, default=np.mean
            Function to summarize the distance metrics.
        random_state : int, default=42
            Seed for reproducible bootstrapping.
        drift_limit : Union[str, Tuple[float, float]], default='stddev'
            Drift threshold definition. Use 'stddev' for automatic thresholds or
            provide custom (lower, upper) bounds.
        method : str, default='expanding'
            Comparison method to use:
            - 'expanding': Each point compared against all accumulated past data
            - 'rolling': Each point compared against a fixed-size rolling window
            - 'jackknife': Each point compared against all other points (leave-one-out)
        window_size : int, optional
            Size of the rolling window when method='rolling'. Required for rolling method.

        Attributes
        ----------
        func : Callable
            The distance function used for drift calculation.
        reference_distribution : np.ndarray
            Normalized probability distribution of reference categories.
        reference_distance : pd.Series
            Calculated distances between reference periods.
        method : str
            The comparison method being used.
        window_size : int
            The window size for rolling method.
        """
        self.method = method
        self.window_size = window_size

        # Validate method and window_size
        if method not in ["expanding", "rolling", "jackknife"]:
            raise ValueError(
                f"method must be one of ['expanding', 'rolling', 'jackknife'], got '{method}'"
            )

        if method == "rolling" and window_size is None:
            raise ValueError("window_size is required when method='rolling'")

        if method == "rolling" and window_size < 2:
            raise ValueError("window_size must be >= 2 for rolling method")

        self.func = self._selection_function(func)

        frequency = self._calculate_frequency(
            X,
        )

        self.reference_distribution = frequency.sum(axis=0) / np.sum(
            frequency.sum(axis=0)
        )

        self.reference_distance = self._generate_distance(
            frequency,
        )

        super().__init__(
            self.reference_distance,
            drift_limit,
        )

    def _calculate_frequency(
        self,
        X: Union[pd.Series, List[np.ndarray], List[list]],
    ) -> pd.DataFrame:
        """
        Calculates the percent distribution of a categorical column grouped by a specified time period.
        """
        index = self._get_index(X)
        X = np.asanyarray(X)
        freq = [Counter(item) for item in X]
        categories = np.unique(np.concatenate(X))
        return pd.DataFrame(freq, columns=categories, index=index)

    def _selection_function(self, func_name: str) -> Callable:
        """Returns a specific function based on the given function name."""

        if func_name == "chebyshev":
            selected_func = chebyshev
        elif func_name == "jensenshannon":
            selected_func = jensenshannon
        elif func_name == "psi":
            selected_func = psi
        else:
            raise ValueError(f"Unsupported distance function: {func_name}")
        return selected_func

    def _generate_distance(
        self,
        X: Union[pd.Series, List[np.ndarray], List[list]],
    ) -> pd.Series:
        """
        Compute a distance metric using different comparison strategies.

        - **Expanding window (method='expanding')**:
            Each point is compared against all accumulated past data.
            Best for detecting gradual drift over time. Efficient O(n).

        - **Rolling window (method='rolling')**:
            Each point is compared against a fixed-size window of past data.
            Good for detecting recent drift while being less sensitive to older data.

        - **Jackknife (method='jackknife')**:
            Each point is compared against all other points (leave-one-out).
            Better for detecting point anomalies. Computationally intensive O(n²).

        Parameters
        ----------
        X : Union[pd.Series, List[np.ndarray], List[list]]
            Frequency counts of categories per period. Rows = time periods,
            columns = categories.

        Returns
        -------
        pd.Series
            Distance metrics indexed by time period. Note:
            - Expanding: First period is dropped (no reference)
            - Rolling: First (window_size-1) periods are dropped
            - Jackknife: All periods included
        """
        index = self._get_index(X)
        X = np.asarray(X)

        if self.method == "expanding":
            return self._expanding_distance(X, index)
        elif self.method == "rolling":
            return self._rolling_distance(X, index)
        elif self.method == "jackknife":
            return self._jackknife_distance(X, index)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _expanding_distance(self, X: np.ndarray, index) -> pd.Series:
        """Compute distances using expanding window approach."""
        n = len(X)
        distances = np.zeros(n)

        past_value = np.zeros(X.shape[1], dtype=np.float64)
        for i in range(1, n):
            past_value = past_value + X[i - 1]
            past_value_norm = past_value / np.sum(past_value)
            current_value_norm = X[i] / np.sum(X[i])
            distances[i] = self.func(past_value_norm, current_value_norm)

        return pd.Series(distances[1:], index=index[1:])

    def _rolling_distance(self, X: np.ndarray, index) -> pd.Series:
        """Compute distances using rolling window approach."""
        n = len(X)
        distances = np.zeros(n)

        for i in range(X.shape[0] - self.window_size + 1):
            past_data = X[i : i + self.window_size]
            past_value = past_data.sum(axis=0)
            past_value_norm = past_value / np.sum(past_value)

            current_value_norm = X[i] / np.sum(X[i])
            distances[i] = self.func(past_value_norm, current_value_norm)

        return pd.Series(distances[self.window_size :], index=index[self.window_size :])

    def _jackknife_distance(self, X: np.ndarray, index) -> pd.Series:
        """Compute distances using jackknife (leave-one-out) approach."""
        n = len(X)
        distances = np.zeros(n)

        for i in range(n):
            current_value_norm = X[i] / np.sum(X[i])
            past_value = np.delete(X, i, axis=0)
            past_value_norm = past_value.sum(axis=0) / np.sum(past_value.sum(axis=0))
            distances[i] = self.func(past_value_norm, current_value_norm)

        return pd.Series(distances, index=index)

    def score(
        self,
        X: Union[pd.Series, List[np.ndarray], List[list]],
    ) -> pd.Series:
        """
        Compute the drift metric between the reference distribution and new data points.
        """
        freq = self._calculate_frequency(X)
        percent = freq.div(freq.sum(axis=1), axis=0)

        return percent.apply(
            lambda row: self.func(row, self.reference_distribution), axis=1
        )
