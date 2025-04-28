import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from .base import BaseModel
from typing import Callable, Tuple, Union


class ContinuousDriftTracker(BaseModel):
    def __init__(
        self,
        reference: Union[pd.Series, pd.core.groupby.SeriesGroupBy, list, np.ndarray],
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
        a reference dataset to compute a baseline distribution and compares subsequent data
        for deviations using statistical distance metrics such as the Wasserstein distance
        or the Kolmogorov-Smirnov test.

        Parameters
        ----------
        reference : Union[pd.Series, pd.core.groupby.SeriesGroupBy, list, np.ndarray]
            The reference dataset used to compute the baseline distribution.
        func : str, optional
            The distance function to use ('ws' for Wasserstein distance or 'ks' for Kolmogorov-Smirnov test).
            Default is 'ws'.
        statistic : callable, optional
            The statistic function used to summarize the reference distance metrics.
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
            the standard deviation of the reference metrics. If a tuple, it specifies custom
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
        self.reference_distribution = reference
        self.reference_distance = self._generate_distance(reference, func)

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
        p: pd.Series,
        func_name: Callable,
    ) -> pd.DataFrame:
        """
        Compute a distance metric (e.g., Kolmogorov-Smirnov test) over a rolling cumulative window.

        This method calculates a specified statistical distance metric between the cumulative
        distribution of past values and the current distribution for each period in the input series.

        Parameters
        p : pd.Series
        func_name : Callable
            A function or callable that computes the distance metric between two distributions.

        Returns
        pd.DataFrame
            A DataFrame containing:
            - 'datetime': The datetime indices corresponding to each period (excluding the first).
            - 'metric': The calculated distance metric for each period.
        """
        func = self._selection_function(func_name)

        n = p.shape[0]
        values = np.zeros(n)
        past_values = np.array([], dtype=float)
        index = p.index[1:]
        p = np.asarray(p)

        for i in range(1, n):
            past_values = np.concatenate([past_values, p[i - 1]])
            value = func(past_values, p[i])
            values[i] = value

        return pd.Series(values[1:], index=index)

    def score(
        self,
        analysis: pd.DataFrame,
    ) -> pd.Series:
        """
        Calculate the drift metric for each time period in the provided dataset.

        Parameters
        ----------
        analysis : pd.DataFrame
            A DataFrame where each row represents a time period and columns represent
            categorical values. The dataset is compared to the reference distribution
            to evaluate drift.

        Returns
        -------
        pd.Series
            A Series containing the calculated drift metric for each time period.
        """
        reference = np.concatenate(np.asarray(self.reference_distribution))
        func = self._selection_function(self.func)

        return analysis.map(lambda row: func(reference, row))
