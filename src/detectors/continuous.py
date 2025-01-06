import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from ..base.model import BaseModel
from typing import Callable, Tuple, Union


class ContinuousDriftDetector(BaseModel):
    def __init__(
        self,
        reference: pd.DataFrame,
        target_col: str,
        datetime_col: str,
        period: str,
        func: str = "ws",
        statistic: Callable = np.mean,
        confidence_level: float = 0.997,
        n_resamples: int = 1000,
        random_state: int = 42,
        drift_limit: Union[str, Tuple[float, float]] = "deviation",
    ):
        """
        A detector for identifying drift in continuous data over time. The detector uses
        a reference dataset to compute a baseline distribution and compare subsequent data
        for deviations using the Kolmogorov-Smirnov test and statistical thresholds.

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
        statistic : callable, optional
            The statistic function used to summarize the reference KS metrics.
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
        thresholds : tuple, optional
            User-defined thresholds for drift detection.
            Default is an empty tuple.

        Attributes:
        ----------
        period : str
            The grouping frequency used for analysis.
        reference_distribution : Series
            The distribution of the reference dataset grouped by the specified period.
        reference_ks : DataFrame
            The Kolmogorov-Smirnov test results for the reference dataset.
        statistics : dict
            Statistical thresholds and summary statistics for drift detection.
        plot : Plot
            A plotting utility for visualizing drift results.
        """

        self._validate_columns(reference, target_col, datetime_col)
        self._validate_params(confidence_level, n_resamples, period)

        self.period = period
        self.func = func

        # Initialize frequency and statistics
        self.reference_distribution = self._calculate_distribution(
            reference,
            target_col,
            datetime_col,
            period,
        )

        self.reference_distance = self._generate_distance(
            self.reference_distribution, func
        )

        super().__init__(
            self.reference_distance,
            confidence_level,
            statistic,
            n_resamples,
            random_state,
            drift_limit,
        )

    def _calculate_distribution(
        self,
        df: pd.DataFrame,
        column_name: str,
        timestamp: str,
        period: str,
    ) -> pd.Series:
        """
        Calculate the continuous distribution of a target column grouped by a given period.

        Parameters:
        ----------
        df : pd.DataFrame
            The dataset to analyze.
        column_name : str
            The name of the column containing the continuous variable.
        timestamp : str
            The name of the datetime column for temporal grouping.
        period : str
            The frequency for grouping (e.g., '1D', '1H').

        Returns:
        -------
        pd.Series
            A Pandas Series where each index corresponds to a time period, and each value is
            a list of continuous values for that period.
        """
        return (
            df[[timestamp, column_name]]
            .copy()
            .groupby(pd.Grouper(key=timestamp, freq=period))[column_name]
            .agg(list)
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
        Calculate the Kolmogorov-Smirnov test metric over a rolling cumulative window.

        Parameters:
        ----------
        p : Series
            A Pandas Series where each element is a list representing the distribution
            of values for a specific period.

        Returns:
        -------
        DataFrame
            A DataFrame containing datetime indices and the calculated KS test metric
            for each period.
        """
        func = self._selection_function(func_name)

        n = p.shape[0]
        values = np.zeros(n)
        past_values = np.array([], dtype=float)

        for i in range(1, n):
            past_values = np.concatenate([past_values, p[i - 1]])
            value = func(past_values, p[i])
            values[i] = value

        return pd.DataFrame({"datetime": p.index[1:], "metric": values[1:]})

    def score(
        self,
        analysis: pd.DataFrame,
        target_col: str,
        datetime_col: str,
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

        self._validate_columns(analysis, target_col, datetime_col)

        reference = np.concatenate(self.reference_distribution)
        dist = self._calculate_distribution(
            analysis, target_col, datetime_col, self.period
        )

        func = self._selection_function(self.func)
        metrics = np.array([func(reference, row) for row in dist])
        metrics = pd.DataFrame(
            {
                "datetime": dist.index,
                "metric": metrics,
            },
        )
        metrics["is_drifted"] = self.is_drifted(metrics)
        return metrics
