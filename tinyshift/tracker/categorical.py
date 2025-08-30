# Copyright (c) 2024-2025 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from .base import BaseModel
from typing import Callable, Tuple, Union, List
from collections import Counter


def l_infinity(a, b):
    """
    Compute the L-infinity distance between two distributions.
    """
    return np.max(np.abs(a - b))


def psi(observed, expected, epsilon=1e-4):
    """
    Calculate Population Stability Index (PSI) between two distributions.
    """
    observed = np.clip(observed, epsilon, 1)
    expected = np.clip(expected, epsilon, 1)
    return np.sum((observed - expected) * np.log(observed / expected))


class CategoricalDriftTracker(BaseModel):
    def __init__(
        self,
        X: Union[pd.Series, List[np.ndarray], List[list]],
        func: str = "l_infinity",
        statistic: Callable = np.mean,
        confidence_level: float = 0.997,
        n_resamples: int = 1000,
        random_state: int = 42,
        drift_limit: Union[str, Tuple[float, float]] = "stddev",
        confidence_interval: bool = False,
        cumulative: bool = True,
    ):
        """
        A tracker for identifying drift in categorical data over time. The tracker uses
        a X dataset to compute a baseline distribution and compares subsequent data
        for deviations based on a distance metric and drift limits.

        Available distance metrics:
        - 'l_infinity': Maximum absolute difference between category probabilities
        - 'jensenshannon': Jensen-Shannon divergence (symmetric, sqrt of JS distance)
        - 'psi': Population Stability Index (sensitive to small probability changes)

        Parameters
        ----------
        X : Union[pd.Series, List[np.ndarray], List[list]]
            Input categorical data. For time series, each element represents a period's
            categorical observations.
        func : str, optional
            Distance metric: 'l_infinity' (default), 'jensenshannon', or 'psi'.
            Default is 'l_infinity'.
        statistic : Callable, optional
            Statistic function to summarize the distance metrics (e.g., np.mean, np.median).
            Default is np.mean.
        confidence_level : float, optional
            Confidence level for statistical thresholds (e.g., 0.997 for 3σ).
            Default is 0.997.
        n_resamples : int, optional
            Number of resamples for bootstrapping when calculating statistics.
            Default is 1000.
        random_state : int, optional
            Seed for reproducible bootstrapping.
            Default is 42.
        drift_limit : Union[str, Tuple[float, float]], optional
            Drift threshold definition:
            - 'stddev': thresholds based on standard deviation of reference metrics
            - tuple: custom (lower, upper) thresholds
            Default is 'stddev'.
        confidence_interval : bool, optional
            Whether to compute bootstrap CIs.
            Default is False.
        cumulative : bool, optional
            - True (cumulative): Aggregates past data for each comparison. Better for gradual drift
              and noisy data. Computationally efficient (O(n)).
            - False (jackknife): Leave-one-out approach. Better for point anomalies but
              computationally intensive (O(n²)).
            Default is True.

        Attributes
        ----------
        func : Callable
            The distance function used for drift calculation.
        reference_distribution : np.ndarray
            Normalized probability distribution of reference categories
        reference_distance : pd.Series
            Calculated distances between reference periods
        """
        self.cumulative = cumulative
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
            confidence_level,
            statistic,
            n_resamples,
            random_state,
            drift_limit,
            confidence_interval,
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

        if func_name == "l_infinity":
            selected_func = l_infinity
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
        Compute a distance metric over a rolling cumulative window or using a jackknife approach.

        - **Cumulative mode (cumulative=True)**:
            For each point, compares it against *all past data* (aggregated up to that point).
            Best for detecting gradual drift over time.

        - **Jackknife mode (cumulative=False)**:
            For each point, compares it against *all other points* (leave-one-out approach).
            This provides a more isolated measure of drift at each timestep but is computationally
            more intensive. Useful for detecting point-wise anomalies or when independence between
            periods is assumed.

        Parameters
        ----------
        X : Union[pd.Series, List[np.ndarray], List[list]]
            Frequency counts of categories per period. Rows = time periods,
            columns = categories.

        Returns
        -------
        pd.Series
            Distance metrics indexed by time period. Note:
            - Cumulative mode: First period is dropped (no reference)
            - Jackknife mode: All periods included
        """
        n = len(X)
        distances = np.zeros(n)
        index = self._get_index(X)
        X = np.asarray(X)

        if self.cumulative:
            past_value = np.zeros(X.shape[1], dtype=np.int32)
            for i in range(1, n):
                past_value = past_value + X[i - 1]
                past_value = past_value / np.sum(past_value)
                current_value = X[i] / np.sum(X[i])
                dist = self.func(past_value, current_value)
                distances[i] = dist
            return pd.Series(distances[1:], index=index[1:])

        for i in range(n):
            current_value = X[i] / np.sum(X[i])
            past_value = np.delete(X, i, axis=0)
            past_value = past_value.sum(axis=0) / np.sum(past_value.sum(axis=0))
            distances[i] = self.func(
                past_value,
                current_value,
            )

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
