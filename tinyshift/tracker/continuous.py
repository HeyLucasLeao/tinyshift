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
        df: pd.DataFrame,
        freq: str = None,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        func: str = "ws",
        drift_limit: Union[str, Tuple[float, float]] = "auto",
        method: str = "expanding",
    ):
        """
        A Tracker for identifying drift in continuous data over time using statistical distance metrics.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing time series data with multiple entities.
        freq : str, optional
            Frequency string for time grouping (e.g., 'D', 'W', 'M'). Default is None.
        id_col : str, default='unique_id'
            Column name containing entity identifiers.
        time_col : str, default='ds'
            Column name containing timestamps.
        target_col : str, default='y'
            Column name containing the target continuous values.
        func : str, default='ws'
            Distance function: 'ws' (Wasserstein distance).
        drift_limit : str or tuple, default='stddev'
            Drift threshold definition:
            - 'stddev': thresholds based on standard deviation of reference metrics
            - tuple: custom (lower, upper) thresholds
        method : str, default='expanding'
            Comparison method to use:
            - 'expanding': Each point compared against all accumulated past data
            - 'jackknife': Each point compared against all other points (leave-one-out)

        Attributes
        ----------
        reference_distribution : pd.DataFrame
            The reference dataset used as baseline.
        reference_distance : pd.Series
            Calculated distance metrics for the reference dataset.
        func : Callable
            The selected distance function.
        method : str
            The comparison method being used.
        freq : str
            The frequency string for time grouping.
        """
        self.method = method
        self.freq = freq

        if self.freq is None:
            raise ValueError("freq must be specified for time grouping.")

        if method not in ["expanding", "jackknife"]:
            raise ValueError(
                f"method must be one of ['expanding', 'jackknife'], got '{method}'"
            )

        self.func = func
        self.func = self._selection_function(func)
        self.reference_distribution = df.groupby(
            [id_col, pd.Grouper(key=time_col, freq=self.freq)]
        )[target_col].apply(np.asarray)

        self.reference_distance = self._generate_distance(self.reference_distribution)

        super().__init__(
            self.reference_distance,
            drift_limit,
            id_col,
        )

    def _wasserstein(self, a: np.ndarray, b: np.ndarray) -> float:
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
        elif self.method == "jackknife":
            return self._jackknife_distance(X, index)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _expanding_distance(self, X: np.ndarray, index) -> pd.Series:
        """Compute distances using expanding window approach."""
        distances = np.zeros(X.shape[0])

        past_value = np.array([], dtype=float)
        for i in range(1, X.shape[0]):
            past_value = np.concatenate([past_value, X[i - 1]])
            distances[i] = self.func(past_value, X[i])

        return pd.Series(distances, index=index)

    def _jackknife_distance(self, X: np.ndarray, index) -> pd.Series:
        """Compute distances using jackknife (leave-one-out) approach."""
        distances = np.zeros(X.shape[0])

        for i in range(X.shape[0]):
            past_value = np.concatenate(np.delete(np.asarray(X), i, axis=0))
            distances[i] = self.func(past_value, X[i])

        return pd.Series(distances, index=index)

    def score(
        self,
        df: pd.DataFrame,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> pd.Series:
        """
        Compute the drift metric between the reference distribution and new data points.
        """
        grouped_data = df.groupby([id_col, pd.Grouper(key=time_col, freq=self.freq)])[
            target_col
        ].apply(np.asarray)

        results = []
        for unique_id in grouped_data.index.get_level_values(0).unique():
            id_data = grouped_data.loc[unique_id]
            reference_data = self.reference_distribution.loc[unique_id]
            reference_combined = np.concatenate(reference_data.values)

            distances = np.array(
                [
                    self.func(current_data, reference_combined)
                    for current_data in id_data.values
                ]
            )

            result_df = pd.DataFrame(
                {
                    id_col: unique_id,
                    time_col: id_data.index,
                    "metric": distances,
                }
            )
            results.append(result_df)

        return pd.concat(results, ignore_index=True)
