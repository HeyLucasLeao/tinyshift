# Copyright (c) 2024-2025 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from .base import BaseModel
from typing import Callable, Tuple, Union, List


class ContinuousDriftTracker(BaseModel):
    def __init__(
        self,
        X: Union[pd.Series, List[np.ndarray], List[list]],
        func: str = "ws",
        statistic: Callable = np.mean,
        confidence_level: float = 0.997,
        n_resamples: int = 1000,
        random_state: int = 42,
        drift_limit: Union[str, Tuple[float, float]] = "stddev",
        confidence_interval: bool = False,
        cumulative: bool = True,
    ):
        """
        A Tracker for identifying drift in continuous data over time using statistical distance metrics.

        Parameters
        ----------
        X : Union[pd.Series, List[np.ndarray], List[list]]
            Input continuous data. For time series, each element represents a period's
            continuous observations.
        func : str, optional
            Distance function: 'ws' (default) or 'ks'.
        statistic : callable, optional
            Statistic function to summarize the distance metrics (e.g., np.mean, np.median).
            Default is np.mean.
        confidence_level : float, optional
            Confidence level for statistical thresholds (e.g., 0.997 for 3σ).
            Default is 0.997.
        n_resamples : int, optional
            Number of resamples for bootstrapping when calculating statistics.
            Default is 1000.
        random_state : int, optional
            Seed for reproducible resampling.
            Default is 42.
        drift_limit : str or tuple, optional
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
        reference_distribution : ArrayLike
            The reference dataset used as baseline.
        reference_distance : pd.Series
            Calculated distance metrics for the reference dataset.
        func : Callable
            The selected distance function (_wasserstein or _ks).
        """
        self.cumulative = cumulative
        self.func = func
        self.func = self._selection_function(func)
        self.reference_distribution = X
        self.reference_distance = self._generate_distance(X)

        super().__init__(
            self.reference_distance,
            confidence_level,
            statistic,
            n_resamples,
            random_state,
            drift_limit,
            confidence_interval,
        )

    def _ks(self, a, b):
        """Calculate the Kolmogorov-Smirnov test and return the p_value."""
        _, p_value = ks_2samp(a, b)
        return p_value

    def _wasserstein(self, a, b):
        """Calculate the Wasserstein Distance."""
        return wasserstein_distance(a, b)

    def _selection_function(self, func_name: str) -> Callable:
        """Returns a specific function based on the given function name."""

        if func_name == "ws":
            selected_func = self._wasserstein
        elif func_name == "ks":
            selected_func = self._ks
        else:
            raise ValueError(f"Unsupported function: {func_name}")
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
            Input data to compute distances. If Series, uses its index for the output.

        Returns
        -------
        pd.Series
            Series with the calculated distances. The index matches the input (excluding the first
            point in cumulative mode).
        """
        n = len(X)
        distances = np.zeros(n)
        index = self._get_index(X)
        X = np.asarray(X)

        if self.cumulative:
            past_value = np.array([], dtype=float)
            for i in range(1, n):
                past_value = np.concatenate([past_value, X[i - 1]])
                value = self.func(past_value, X[i])
                distances[i] = value
            return pd.Series(distances[1:], index=index[1:])

        for i in range(n):
            past_value = np.concatenate(np.delete(np.asarray(X), i, axis=0))
            distances[i] = self.func(
                past_value,
                X[i],
            )

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
