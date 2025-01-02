import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import f1_score


class PerformanceTracker:
    """
    Class to monitor model performance over time periods and detect anomalies
    using metrics, confidence intervals, and thresholds.

    Methods:
    --------
    - fit: Calculate reference metrics grouped by a time period.
    - metric_by_time_period: Calculate a metric for each time period in the DataFrame.
    - bootstrapping_bca: Calculate BCa bootstrap confidence intervals.
    """

    def __init__(self):
        """
        Initialize the MonitorModel class.
        """
        self.reference_metrics = None

    def metric_by_time_period(
        self, df, period, target_col, prediction_col, metric=f1_score
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
        grouped = df.groupby(pd.Grouper(key="datetime", freq=period)).apply(
            lambda x: metric(x[target_col], x[prediction_col])
        )
        return grouped.reset_index(name="metric")

    def bootstrapping_bca(
        self,
        data,
        confidence_level=0.997,
        statistic=np.mean,
        n_resamples=1000,
        random_state=42,
    ):
        """
        Calculate BCa bootstrap confidence intervals.

        Parameters:
        - data: List or numpy array with sample data.
        - confidence_level: Desired confidence level (default 0.997).
        - statistic: Statistical function to apply to the data (default np.mean).
        - n_resamples: Number of bootstrap resamples (default 1000).
        - random_state: Random seed for reproducibility.

        Returns:
        - tuple: Lower and upper bounds of the BCa confidence interval.
        """
        np.random.seed(random_state)
        data = np.asarray(data)
        n = len(data)

        def generate_acceleration(data):
            jackknife = np.zeros(n)
            for i in range(n):
                jackknife_sample = np.delete(data, i)
                jackknife[i] = statistic(jackknife_sample)

            jackknife_mean = np.mean(jackknife)
            jackknife_diffs = jackknife - jackknife_mean
            acceleration = np.sum(jackknife_diffs**3) / (
                6.0 * (np.sum(jackknife_diffs**2) ** 1.5)
            )
            return acceleration

        def calculate_bootstrap_statistics(data, statistic, n_resamples):
            return np.array(
                [
                    statistic(np.random.choice(data, size=n, replace=True))
                    for _ in range(n_resamples)
                ]
            )

        sample_statistics = calculate_bootstrap_statistics(data, statistic, n_resamples)
        acceleration = generate_acceleration(data)
        observed_stat = statistic(data)
        bias = np.mean(sample_statistics < observed_stat)
        z0 = norm.ppf(bias)

        alpha = 1 - confidence_level
        z_alpha = norm.ppf(1 - alpha / 2)

        z_lower_bound = (z0 - z_alpha) / (1 - acceleration * (z0 - z_alpha)) + z0
        z_upper_bound = (z0 + z_alpha) / (1 - acceleration * (z0 + z_alpha)) + z0

        alpha_lower = norm.cdf(z_lower_bound)
        alpha_upper = norm.cdf(z_upper_bound)

        lower_bound = np.quantile(sample_statistics, alpha_lower)
        upper_bound = np.quantile(sample_statistics, alpha_upper)

        return lower_bound, upper_bound

    def fit(
        self,
        df,
        period,
        target_col,
        prediction_col,
        score_method,
        statistic=np.mean,
        confidence_level=0.997,
        n_resamples=1000,
        random_state=42,
    ):
        """
        Calculate reference metrics grouped by a time period, including confidence intervals and thresholds.

        Parameters:
        - df: Input DataFrame containing data.
        - period: Time period for grouping (e.g., 'W' for weeks).
        - target_col: Column name for actual target values.
        - prediction_col: Column name for predicted values.
        - score_method: Metric function for evaluation (e.g., accuracy, F1 score).
        - statistic: Statistical function to apply to the metric (e.g., np.mean).
        - confidence_level: Confidence level for BCa intervals (default 0.997).
        - n_resamples: Number of bootstrap resamples (default 1000).
        - random_state: Random seed (default 42).

        Returns:
        - dict: Reference metrics including confidence intervals and thresholds.
        """
        metrics_by_period = self.metric_by_time_period(
            df, period, target_col, prediction_col, score_method
        )

        ci_lower, ci_upper = self.bootstrapping_bca(
            metrics_by_period["metric"],
            confidence_level,
            statistic,
            n_resamples,
            random_state,
        )

        estimated_mean_statistic = np.mean(metrics_by_period["metric"])
        std_deviation = metrics_by_period["metric"].std()
        lower_threshold = estimated_mean_statistic - (std_deviation * 3)
        upper_threshold = estimated_mean_statistic + (std_deviation * 3)

        self.reference_metrics = {
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "mean": estimated_mean_statistic,
            "lower_threshold": lower_threshold,
            "upper_threshold": upper_threshold,
        }

        return self.reference_metrics
