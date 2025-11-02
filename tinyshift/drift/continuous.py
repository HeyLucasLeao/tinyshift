# Copyright (c) 2024-2025 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from .base import BaseModel
from typing import Callable, Tuple, Union, List
from sklearn.base import BaseEstimator


class ConDrift(BaseModel, BaseEstimator):
    """
    A tracker for identifying drift in continuous data over time.

    The tracker uses a reference dataset to compute a baseline distribution and compares
    subsequent data for deviations based on a distance metric and drift limits.

    Available distance metrics:
    - 'ws': Wasserstein distance (Earth Mover's Distance) - measures the minimum cost
      to transform one distribution into another

    Comparison methods:
    - 'expanding': Each point compared against all accumulated past data
    - 'jackknife': Each point compared against all other points (leave-one-out)

    Attributes
    ----------
    func : Callable
        The distance function used for drift calculation.
    reference_distribution : dict
        Dictionary mapping unique_id to reference data arrays used as baseline.
    method : str
        The comparison method being used.
    freq : str
        The frequency parameter for time grouping.
    """

    def __init__(
        self,
        freq: str = None,
        func: str = "ws",
        drift_limit: Union[str, Tuple[float, float]] = "auto",
        method: str = "expanding",
    ):
        """
        Initialize the continuous drift detector.

        Parameters
        ----------
        freq : str
            Frequency for time grouping (e.g., 'D', 'W', 'M'). Required for time-based analysis.
        func : str, default='ws'
            Distance metric to use for drift detection. Options: 'ws' (Wasserstein distance).
        drift_limit : Union[str, Tuple[float, float]], default='auto'
            Drift threshold definition. Use 'auto' for automatic thresholds or
            provide custom (lower, upper) bounds.
        method : str, default='expanding'
            Comparison method:
            - 'expanding': Each point compared against accumulated past data
            - 'jackknife': Each point compared against all other points (leave-one-out)
        """

        if freq is None:
            raise ValueError("freq must be specified for time grouping.")

        if method not in ["expanding", "jackknife"]:
            raise ValueError(
                f"method must be one of ['expanding', 'jackknife'], got '{method}'"
            )

        self.freq = freq
        self.func = self._selection_function(func)
        self.drift_limit = drift_limit
        self.method = method
        self.reference_distribution = None

    def fit(
        self,
        df: pd.DataFrame,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> "ConDrift":
        """
        Fit the drift detector to reference data.

        Parameters
        ----------
        df : pd.DataFrame
            Reference dataframe containing continuous data with time series structure.
        id_col : str, default='unique_id'
            Column name identifying unique time series entities.
        time_col : str, default='ds'
            Column name containing timestamps for time-based grouping.
        target_col : str, default='y'
            Column name containing continuous values to analyze for drift.

        Returns
        -------
        self : ConDrift
            Returns self for method chaining.
        """
        self._check_dataframe(df, time_col, target_col, id_col)

        reference = df.groupby([id_col, pd.Grouper(key=time_col, freq=self.freq)])[
            target_col
        ].apply(np.asarray)

        reference_distance = self._generate_distance(reference)

        self.reference_distribution = {
            unique_id: np.concatenate(reference.loc[unique_id].values).astype(
                np.float32
            )
            for unique_id in reference.index.get_level_values(0).unique()
        }

        super().__init__(
            reference_distance,
            self.drift_limit,
            id_col,
        )

        return self

    def _selection_function(self, func_name: str) -> Callable:
        """Returns a specific function based on the given function name."""

        if func_name == "ws":
            selected_func = wasserstein_distance
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
        self._check_dataframe(df, time_col, target_col, id_col)

        grouped_data = df.groupby([id_col, pd.Grouper(key=time_col, freq=self.freq)])[
            target_col
        ].apply(np.asarray)

        results = []
        for unique_id in grouped_data.index.get_level_values(0).unique():
            id_data = grouped_data.loc[unique_id]
            reference_data = self.reference_distribution[unique_id]

            distances = np.array(
                [
                    self.func(current_data, reference_data)
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
