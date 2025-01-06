import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from ..base.model import BaseModel
from typing import Callable, Tuple, Union


class ContinuousDriftDetector(BaseModel):
    def __init__(
        self,
        reference: pd.DataFrame,
        target_col: str,
        datetime_col: str,
        period: str,
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

        if target_col not in reference.columns:
            raise KeyError(f"Column {target_col} is not in the DataFrame.")
        if datetime_col not in reference.columns:
            raise KeyError(f"Datetime column {datetime_col} is not in the DataFrame.")
        if not pd.api.types.is_datetime64_any_dtype(reference[datetime_col]):
            raise TypeError(f"Column {datetime_col} must be of datetime type.")

        self.period = period
        # Initialize frequency and statistics
        self.reference_distribution = self._calculate_distribution(
            reference,
            target_col,
            datetime_col,
            period,
        )

        self.reference_ks = self._generate_ks(
            self.reference_distribution,
        )

        super().__init__(
            self.reference_ks,
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

    def _generate_ks(
        self,
        p: pd.Series,
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
        n = p.shape[0]
        p_values = np.zeros(n)
        past_values = np.array([], dtype=float)

        for i in range(1, n):
            past_values = np.concatenate([past_values, p[i - 1]])
            _, p_value = ks_2samp(past_values, p[i])
            p_values[i] = p_value

        return pd.DataFrame({"datetime": p.index[1:], "metric": p_values[1:]})

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

        if target_col not in analysis.columns:
            raise KeyError(f"Column {target_col} is not in the DataFrame.")
        if datetime_col not in analysis.columns:
            raise KeyError(f"Datetime column {datetime_col} is not in the DataFrame.")
        if not pd.api.types.is_datetime64_any_dtype(analysis[datetime_col]):
            raise TypeError(f"Column {datetime_col} must be of datetime type.")

        reference = np.concatenate(self.reference_distribution)
        dist = self._calculate_distribution(
            analysis, target_col, datetime_col, self.period
        )

        metrics = np.array([ks_2samp(reference, row)[1] for row in dist])

        return pd.DataFrame(
            {
                "datetime": dist.index,
                "metric": metrics,
                "is_drifted": metrics <= self.statistics["lower_limit"],
            },
        )
