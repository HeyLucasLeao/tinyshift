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
    ):
        """
        A Tracker for identifying drift in continuous data over time. This tracker uses
        a X dataset to compute a baseline distribution and compares subsequent data
        for deviations using statistical distance metrics such as the Wasserstein distance
        or the Kolmogorov-Smirnov test.

        Parameters
        ----------
        X : Union[pd.Series, pd.core.groupby.SeriesGroupBy, list, np.ndarray]
            The X dataset used to compute the baseline distribution.
        func : str, optional
            The distance function to use ('ws' for Wasserstein distance or 'ks' for Kolmogorov-Smirnov test).
            Default is 'ws'.
        statistic : callable, optional
            The statistic function used to summarize the X distance metrics.
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
        drift_limit : str or tuple, optional
            Defines the threshold for drift detection. If 'stddev', thresholds are based on
            the standard deviation of the X metrics. If a tuple, it specifies custom
            lower and upper thresholds.
            Default is 'stddev'.
        confidence_interval : bool, optional
            Whether to calculate confidence intervals for the drift metrics.
            Default is False.

        Attributes
        ----------
        func : str
            The selected distance function ('ws' or 'ks').
        reference_distribution : Union[pd.Series, pd.core.groupby.SeriesGroupBy, list, np.ndarray]
            The reference dataset used to compute the baseline distribution.
        reference_distance : DataFrame
            The calculated distance metrics for the reference dataset.
        """

        self.func = func
        self.reference_distribution = X
        self.reference_distance = self._generate_distance(X, func)

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
        func_name: Callable,
    ) -> pd.Series:
        """
        Compute a distance metric over a rolling cumulative window.

        This method calculates a specified statistical distance metric (e.g., Kolmogorov-Smirnov test)
        between the cumulative distribution of past values and the current value for each period in
        the input series.

        ----------
        X : Union[pd.Series, List[np.ndarray], List[list]]
            The input data series or list of arrays/lists to compute the distance metric on.

        -------
        pd.Series
            A Series containing:
            - Index: The datetime indices corresponding to each period (excluding the first).
            - Values: The calculated distance metric for each period.
        """
        func = self._selection_function(func_name)

        n = X.shape[0]
        values = np.zeros(n)
        past_values = np.array([], dtype=float)
        index = X.index[1:]
        X = np.asarray(X)

        for i in range(1, n):
            past_values = np.concatenate([past_values, X[i - 1]])
            value = func(past_values, X[i])
            values[i] = value

        return pd.Series(values[1:], index=index)

    def score(
        self,
        X: Union[pd.Series, List[np.ndarray], List[list]],
    ) -> pd.Series:
        """
        Compute the drift metric for each time period in the provided dataset.
        """
        reference = np.concatenate(np.asarray(self.reference_distribution))
        func = self._selection_function(self.func)
        index = self._get_index(X)
        X = np.asarray(X)

        return pd.Series([func(reference, row) for row in X], index=index)
