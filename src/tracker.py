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
        timestamp_col,
        period,
        metric_score=f1_score,
        statistic=np.mean,
        confidence_level=0.997,
        n_resamples=1000,
        random_state=42,
    ):

        # Initialize distributions and statistics
        self.reference_distribution = self._calculate_metric(
            reference_data,
            target_col,
            prediction_col,
            timestamp_col,
            period,
            metric_score,
        )
        self.statistics = scoring.calculate_statistics(
            self.reference_distribution,
            confidence_level,
            statistic,
            n_resamples=n_resamples,
            random_state=random_state,
        )
        self.plot = plot.Plot(self.statistics, self.reference_distribution)

    def _calculate_metric(
        self,
        df,
        target_col,
        prediction_col,
        timestamp_col,
        period,
        metric,
    ):
        """
        Calculate a specified metric for each time period in the DataFrame.

        Parameters:
        - df: pandas DataFrame containing the data.
        - period: The time period for grouping data (e.g., 'W' for weeks, 'M' for months).
        - target_col: The name of the column containing the actual target values.
        - prediction_col: The name of the column containing the predicted values.
        - metric: A function to compute the metric (e.g., f1_score).

        Returns:
        - pandas DataFrame: A DataFrame with the metric calculated for each time period.
        """
        grouped = df.groupby(pd.Grouper(key=timestamp_col, freq=period)).apply(
            lambda x: metric(x[target_col], x[prediction_col])
        )
        return grouped.reset_index(name="metric")
