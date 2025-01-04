import numpy as np
from src import scoring
from sklearn.metrics import f1_score


class PerformanceTracker:
    def __init__(
        self,
        reference_data,
        target_col,
        prediction_col,
        timestamp_col,
        period,
        metric_score=f1_score,
        distance_function="l_infinity",
        statistic=np.mean,
        confidence_level=0.997,
        n_resamples=1000,
        random_state=42,
    ):
        self.column_col = target_col
        self.prediction_col = prediction_col
        self.timestamp_col = timestamp_col
        self.period = period
        self.distance_function = distance_function
        self.statistic = statistic
        self.confidence_level = confidence_level
        self.n_resamples = n_resamples
        self.random_state = random_state

        # Initialize distributions and statistics
        self.reference_distribution = scoring.metric_by_time_period(
            reference_data,
            target_col,
            prediction_col,
            timestamp_col,
            period,
            metric_score,
        )
        self.statistics = self._calculate_statistics()

    def _calculate_statistics(self):
        """
        Calculate statistics for the reference distances, including confidence intervals and thresholds.
        """
        ci_lower, ci_upper = scoring.bootstrapping_bca(
            self.reference_distribution["metric"],
            self.confidence_level,
            self.statistic,
            self.n_resamples,
            self.random_state,
        )

        threshold_lower, threshold_upper = scoring.deviation_threshold(
            self.reference_distribution
        )

        estimated_mean = np.mean(self.reference_distribution["metric"])

        return {
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "mean": estimated_mean,
            "lower_threshold": threshold_lower,
            "upper_threshold": threshold_upper,
        }
