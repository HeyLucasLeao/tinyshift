import numpy as np
from . import plot
from src import scoring
from sklearn.metrics import f1_score
import pandas as pd


class PerformanceTracker:
    def __init__(
        self,
        reference_data,
        target_col,
        prediction_col,
        datetime_col,
        period,
        metric_score=f1_score,
        statistic=np.mean,
        confidence_level=0.997,
        n_resamples=1000,
        random_state=42,
        thresholds=(),
    ):
        """
        A tracker for monitoring model performance over time using a specified evaluation metric.
        The tracker compares the performance metric across time periods to a reference distribution
        and identifies potential performance degradation.

        Parameters:
        ----------
        reference_data : DataFrame
            The reference dataset used to compute the baseline metric distribution.
        target_col : str
            The name of the column containing the actual target values.
        prediction_col : str
            The name of the column containing the predicted values.
        datetime_col : str
            The name of the column containing datetime values for temporal grouping.
        period : str
            The frequency for grouping data (e.g., 'W' for weekly, 'M' for monthly).
        metric_score : callable, optional
            The function to compute the evaluation metric (e.g., `f1_score`).
            Default is `f1_score`.
        statistic : callable, optional
            The statistic function used to summarize the reference metric distribution.
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
        metric_score : callable
            The evaluation metric function used for tracking performance.
        reference_distribution : DataFrame
            The performance metric distribution of the reference dataset.
        statistics : dict
            Statistical thresholds and summary statistics for performance monitoring.
        plot : Plot
            A plotting utility for visualizing performance over time.
        """

        self.period = period
        self.metric_score = metric_score

        # Initialize distributions and statistics
        self.reference_distribution = self._calculate_metric(
            reference_data,
            target_col,
            prediction_col,
            datetime_col,
        )
        self.statistics = scoring.calculate_statistics(
            self.reference_distribution,
            confidence_level,
            statistic,
            n_resamples=n_resamples,
            random_state=random_state,
        )
        self.plot = plot.Plot(self.statistics, self.reference_distribution)
        scoring.check_thresholds(
            self.statistics, self.reference_distribution, thresholds
        )

    def _calculate_metric(
        self,
        df,
        target_col,
        prediction_col,
        datetime_col,
    ):
        """
        Calculate the performance metric for each time period in the dataset.

        Parameters:
        ----------
        df : DataFrame
            The dataset containing the data to analyze.
        target_col : str
            The name of the column containing the actual target values.
        prediction_col : str
            The name of the column containing the predicted values.
        datetime_col : str
            The name of the datetime column for temporal grouping.

        Returns:
        -------
        DataFrame
            A DataFrame with the calculated metric for each time period.
        """
        grouped = df.groupby(pd.Grouper(key=datetime_col, freq=self.period)).apply(
            lambda x: self.metric_score(x[target_col], x[prediction_col])
        )
        return grouped.reset_index(name="metric")

    def score(self, df, target_col, prediction_col, datetime_col):
        """
        Assess model performance over time by calculating the evaluation metric
        for each time period and comparing it to the reference distribution.

        Parameters:
        ----------
        df : DataFrame
            The dataset to analyze for performance drift.
        target_col : str
            The name of the column containing the actual target values.
        prediction_col : str
            The name of the column containing the predicted values.
        datetime_col : str
            The name of the datetime column for temporal grouping.

        Returns:
        -------
        DataFrame
            A DataFrame containing datetime values, calculated metrics, and a boolean
            indicating whether performance drift was detected for each time period.
        """
        res = self._calculate_metric(df, target_col, prediction_col, datetime_col)
        res["is_drifted"] = res["metric"] <= self.statistics["lower_threshold"]
        return res
