import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from .base import BaseModel
from typing import Callable, Tuple, Union


class ContinuousDriftTracker(BaseModel):
    def __init__(
        self,
        reference: pd.DataFrame,
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

        Parameters:
        ----------
        reference : DataFrame
            The reference dataset used to compute the baseline distribution.
        target_col : str
            The name of the column containing the continuous variable to analyze.
        datetime_col : str
            The name of the column containing datetime values for temporal grouping.
        period : str
            The frequency for grouping data (e.g., '1D' for daily, '1H' for hourly).
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

        Attributes:
        ----------
        period : str
            The grouping frequency used for analysis.
        func : str
            The selected distance function ('ws' or 'ks').
        reference_distribution : Series
            The distribution of the reference dataset grouped by the specified period.
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

        return pd.DataFrame({"datetime": index, "metric": values[1:]})

    def score(
        self,
        analysis: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Assess drift in the provided dataset by comparing its distribution to the reference.

        Parameters:
        ----------
        analysis : DataFrame
            The dataset to analyze for drift.
        target_col : str
            The name of the continuous column in the analysis dataset.
        datetime_col : str
            The name of the datetime column in the analysis dataset.

        Returns:
        -------
        DataFrame
            A DataFrame containing datetime values, drift metrics, and a boolean
            indicating whether drift was detected for each time period.
        """
        reference = np.concatenate(np.asarray(self.reference_distribution))
        func = self._selection_function(self.func)
        metrics = np.array([func(reference, row) for row in analysis])
        metrics = pd.DataFrame(
            {
                "datetime": analysis.index,
                "metric": metrics,
            },
        )
        metrics["is_drifted"] = self._is_drifted(metrics)
        return metrics
