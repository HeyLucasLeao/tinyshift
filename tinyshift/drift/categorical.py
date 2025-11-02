# Copyright (c) 2024-2025 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from .base import BaseModel
from typing import Callable, Tuple, Union, List
from sklearn.base import BaseEstimator


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


class CatDrift(BaseModel, BaseEstimator):
    """
    A tracker for identifying drift in categorical data over time.

    The tracker uses a reference dataset to compute a baseline distribution and compares
    subsequent data for deviations based on a distance metric and drift limits.

    Available distance metrics:
    - 'chebyshev': Maximum absolute difference between category probabilities
    - 'jensenshannon': Jensen-Shannon divergence (symmetric, sqrt of JS distance)
    - 'psi': Population Stability Index (sensitive to small probability changes)

    Attributes
    ----------
    func : Callable
        The distance function used for drift calculation.
    reference_distribution : dict
        Normalized probability distribution of reference categories.
    method : str
        The comparison method being used.
    freq : str
        The frequency parameter for time grouping.
    """

    def __init__(
        self,
        freq: str = None,
        func: str = "chebyshev",
        drift_limit: Union[str, Tuple[float, float]] = "auto",
        method: str = "expanding",
    ):
        """
        Initialize the categorical drift detector.

        Parameters
        ----------
        freq : str
            Frequency for time grouping (e.g., 'D', 'W', 'M'). Required for time-based analysis.
        func : str, default='chebyshev'
            Distance metric to use for drift detection. Options: 'chebyshev', 'jensenshannon', 'psi'.
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
    ) -> "CatDrift":
        """
        Fit the drift detector to reference data.

        Parameters
        ----------
        df : pd.DataFrame
            Reference dataframe containing categorical data with time series structure.
        id_col : str, default='unique_id'
            Column name identifying unique time series entities.
        time_col : str, default='ds'
            Column name containing timestamps for time-based grouping.
        target_col : str, default='y'
            Column name containing categorical values to analyze for drift.

        Returns
        -------
        self : CatDrift
            Returns self for method chaining.
        """
        self._check_dataframe(df, time_col, target_col, id_col)

        frequency = (
            df.groupby([id_col, pd.Grouper(key=time_col, freq=self.freq), target_col])[
                target_col
            ]
            .size()
            .unstack(fill_value=0)
        )
        reference = frequency.groupby([id_col]).sum() / np.sum(frequency.sum(axis=0))

        reference_distance = self._generate_distance(
            frequency,
        )

        self.reference_distribution = {
            unique_id: {
                category: round(prob, 4)
                for category, prob in reference.loc[unique_id].items()
            }
            for unique_id in reference.index
        }

        super().__init__(
            reference_distance,
            self.drift_limit,
            id_col,
        )

        return self

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
        n = len(X)
        distances = np.zeros(n)

        past_value = np.zeros(X.shape[1], dtype=np.float64)
        for i in range(1, n):
            past_value = past_value + X[i - 1]
            past_value_norm = past_value / np.sum(past_value)
            current_value_norm = X[i] / np.sum(X[i])
            distances[i] = self.func(past_value_norm, current_value_norm)

        return pd.Series(distances, index=index)

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
        df: pd.DataFrame,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> pd.Series:
        """
        Compute the drift metric between the reference distribution and new data points.
        """
        self._check_dataframe(df, time_col, target_col, id_col)

        frequency = (
            df.groupby([id_col, pd.Grouper(key=time_col, freq=self.freq), target_col])[
                target_col
            ]
            .size()
            .unstack(fill_value=0)
        )
        percent = frequency.div(frequency.sum(axis=1), axis=0)

        results = []
        for unique_id in percent.index.get_level_values(0).unique():
            id_data = percent.loc[unique_id]
            reference = self.reference_distribution[unique_id]

            common_cols = id_data.columns.intersection(reference.keys())
            id_data_aligned = id_data[common_cols]
            reference_aligned = np.array([reference[col] for col in common_cols])
            distances = np.array(
                [self.func(row, reference_aligned) for row in id_data_aligned.values]
            )

            result_df = pd.DataFrame(
                {
                    id_col: unique_id,
                    time_col: id_data_aligned.index,
                    "metric": distances,
                }
            )
            results.append(result_df)

        return pd.concat(results, ignore_index=True)
