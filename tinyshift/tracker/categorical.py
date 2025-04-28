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
    ):
        """
        A tracker for identifying drift in categorical data over time. The tracker uses
        a X dataset to compute a baseline distribution and compares subsequent data
        for deviations based on a distance metric and drift limits.

        Parameters
        ----------
        X : pd.DataFrame
            The X dataset used to compute the baseline distribution.
        func : str, optional
            The distance function to use ('l_infinity' or 'jensenshannon').
            Default is 'l_infinity'.
        statistic : Callable, optional
            The statistic function used to summarize the X distances.
            Default is `np.mean`.
        confidence_level : float, optional
            The confidence level for calculating statistical thresholds.
            Default is 0.997.
        n_resamples : int, optional
            Number of resamples for bootstrapping when calculating statistics.
            Default is 1000.
        random_state : int, optional
            Seed for reproducibility of random resampling.
            Default is 42.
        drift_limit : Union[str, Tuple[float, float]], optional
            User-defined thresholds for drift detection. If set to "stddev", thresholds
            are calculated based on the standard deviation of the X distances.
            Default is "stddev".
        confidence_interval : bool, optional
            Whether to calculate and include confidence intervals in the drift analysis.
            Default is False.

        Attributes
        ----------
        func : Callable
            The distance function used for drift calculation.
        reference_distribution : np.ndarray
            The normalized distribution of the reference dataset.
        reference_distance : pd.DataFrame
            The distance metric values for the reference dataset.
        """

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
        else:
            raise ValueError(f"Unsupported distance function: {func_name}")
        return selected_func

    def _generate_distance(
        self,
        X: Union[pd.Series, List[np.ndarray], List[list]],
    ) -> pd.Series:
        """
        Calculates a distance metric between consecutive rows of a frequency
        distribution DataFrame, where rows represent time periods and columns
        represent categorical values. The distance is computed using a specified
        function.

        ----------
        X : Union[pd.Series, List[np.ndarray], List[list]]
            A data structure representing the frequency distribution, where rows
            correspond to time periods and columns correspond to categorical values.

        -------
        pd.Series
            A Series containing the calculated distance metric for each consecutive
            period, indexed by the datetime values corresponding to the time periods
            (excluding the first period).
        """
        n = X.shape[0]
        distances = np.zeros(n)
        past_value = np.zeros(X.shape[1], dtype=np.int32)
        index = X.index[1:]
        X = np.asarray(X)

        for i in range(1, n):
            past_value = past_value + X[i - 1]
            past_value = past_value / np.sum(past_value)
            current_value = X[i] / np.sum(X[i])
            dist = self.func(past_value, current_value)
            distances[i] = dist

        return pd.Series(distances[1:], index=index)

    def score(
        self,
        X: Union[pd.Series, List[np.ndarray], List[list]],
    ) -> pd.Series:
        """
        Compute the drift metric for each time period in the provided dataset.
        """
        freq = self._calculate_frequency(X)
        percent = freq.div(freq.sum(axis=1), axis=0)

        return percent.apply(
            lambda row: self.func(row, self.reference_distribution), axis=1
        )
