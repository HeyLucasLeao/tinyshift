# Copyright (c) 2024-2025 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from .base import BaseModel
from typing import Callable, Tuple, Union, List


class ConDrift(BaseModel):
    def __init__(
        self,
        X: Union[pd.Series, List[np.ndarray], List[list]],
        func: str = "ws",
        statistic: Callable = np.mean,
        random_state: int = 42,
        drift_limit: Union[str, Tuple[float, float]] = "stddev",
        method: str = "expanding",
        window_size: int = None,
    ):
        """
        A Tracker for identifying drift in continuous data over time using statistical distance metrics.

        Parameters
        ----------
        X : Union[pd.Series, List[np.ndarray], List[list]]
            Input continuous data. For time series, each element represents a period's
            continuous observations.
        func : str, optional
            Distance function: 'ws' (Wasserstein distance). Default is 'ws'.
        statistic : callable, optional
            Statistic function to summarize the distance metrics (e.g., np.mean, np.median).
            Default is np.mean.
        random_state : int, optional
            Seed for reproducible resampling. Default is 42.
        drift_limit : str or tuple, optional
            Drift threshold definition:
            - 'stddev': thresholds based on standard deviation of reference metrics
            - tuple: custom (lower, upper) thresholds
            Default is 'stddev'.
        method : str, default='expanding'
            Comparison method to use:
            - 'expanding': Each point compared against all accumulated past data
            - 'rolling': Each point compared against a fixed-size rolling window
            - 'jackknife': Each point compared against all other points (leave-one-out)
        window_size : int, optional
            Size of the rolling window when method='rolling'. Required for rolling method.

        Attributes
        ----------
        reference_distribution : ArrayLike
            The reference dataset used as baseline.
        reference_distance : pd.Series
            Calculated distance metrics for the reference dataset.
        func : Callable
            The selected distance function.
        method : str
            The comparison method being used.
        window_size : int
            The window size for rolling method.
        """
        self.method = method
        self.window_size = window_size

        if method not in ["expanding", "rolling", "jackknife"]:
            raise ValueError(
                f"method must be one of ['expanding', 'rolling', 'jackknife'], got '{method}'"
            )

        if method == "rolling" and window_size is None:
            raise ValueError("window_size is required when method='rolling'")

        if method == "rolling" and window_size < 2:
            raise ValueError("window_size must be >= 2 for rolling method")

        self.func = func
        self.func = self._selection_function(func)
        self.reference_distribution = X
        self.reference_distance = self._generate_distance(X)

        super().__init__(
            self.reference_distance,
            statistic,
            random_state,
            drift_limit,
        )

    def _wasserstein(self, a, b):
        """Calculate the Wasserstein Distance."""
        return wasserstein_distance(a, b)

    def _selection_function(self, func_name: str) -> Callable:
        """Returns a specific function based on the given function name."""

        if func_name == "ws":
            selected_func = self._wasserstein
        else:
            raise ValueError(f"Unsupported function: {func_name}")
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
            Input data to compute distances. If Series, uses its index for the output.

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

        past_value = np.array([], dtype=float)
        for i in range(1, n):
            past_value = np.concatenate([past_value, X[i - 1]])
            distances[i] = self.func(past_value, X[i])

        return pd.Series(distances[1:], index=index[1:])

    def _rolling_distance(self, X: np.ndarray, index) -> pd.Series:
        """Compute distances using rolling window approach."""
        n = len(X)
        distances = np.zeros(n)

        for i in range(X.shape[0] - self.window_size + 1):
            past_data = np.concatenate(X[i : i + self.window_size])
            distances[i] = self.func(past_data, X[i])

        return pd.Series(distances[self.window_size :], index=index[self.window_size :])

    def _jackknife_distance(self, X: np.ndarray, index) -> pd.Series:
        """Compute distances using jackknife (leave-one-out) approach."""
        n = len(X)
        distances = np.zeros(n)

        for i in range(n):
            past_value = np.concatenate(np.delete(np.asarray(X), i, axis=0))
            distances[i] = self.func(past_value, X[i])

        return pd.Series(distances, index=index)

    def score(
        self,
        X: Union[pd.Series, List[np.ndarray], List[list]],
    ) -> pd.Series:
        """
        Compute the drift metric between the reference distribution and new data points.
        """
        reference = np.concatenate(np.asarray(self.reference_distribution))
        index = self._get_index(X)
        X = np.asarray(X)

        return pd.Series([self.func(reference, row) for row in X], index=index)
