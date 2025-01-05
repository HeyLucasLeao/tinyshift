import numpy as np
import pandas as pd
from src import scoring
from scipy.spatial.distance import jensenshannon


def l_infinity(a, b):
    """
    Compute the L-infinity distance between two distributions.
    """
    return np.max(np.abs(a - b))


class CategoricalDriftDetector:
    def __init__(
        self,
        reference_data,
        column_name,
        period,
        column_timestamp,
        distance_function="l_infinity",
        statistic=np.mean,
        confidence_level=0.997,
        n_resamples=1000,
        random_state=42,
    ):
        self.reference_data = reference_data
        self.column_name = column_name
        self.period = period
        self.column_timestamp = column_timestamp
        self.distance_function = distance_function
        self.statistic = statistic
        self.confidence_level = confidence_level
        self.n_resamples = n_resamples
        self.random_state = random_state

        # Initialize frequency and statistics
        self.reference_frequency = self._calculate_frequency(
            reference_data, period, column_timestamp
        )
        self.reference_distance = self.generate_distance(self.reference_frequency)
        self.statistics = self._calculate_statistics()

    def _calculate_frequency(self, df, period, column_timestamp):
        """
        Calculate the reference frequency by grouping data based on the specified period.
        """
        grouped = (
            df.groupby(pd.Grouper(key=column_timestamp, freq=period))
            .apply(lambda x: x[self.column_name].value_counts())
            .rename("metric")
            .reset_index()
            .rename(columns={"level_1": self.column_name})
        )

        grouped = grouped.set_index([column_timestamp, self.column_name]).unstack(
            fill_value=0
        )
        return grouped

    def generate_distance(self, p):
        """
        Generate the distance metric based on the specified distance function.
        """
        n = p.shape[0]
        p_values = np.zeros(n)
        past_value = np.zeros(p.shape[1], dtype=np.int32)
        index = p.index[1:]
        p = np.asarray(p)

        if self.distance_function == "l_infinity":
            func = l_infinity
        elif self.distance_function == "jensenshannon":
            func = jensenshannon
        else:
            raise ValueError(f"Unsupported distance function: {self.distance_function}")

        for i in range(1, n):
            past_value = past_value + p[i - 1] / np.sum(past_value)
            current_value = p[i] / np.sum(p[i])
            p_value = func(past_value, current_value)
            p_values[i] = p_value

        return pd.DataFrame({"datetime": index, "metric": p_values[1:]})

    def _calculate_statistics(self):
        """
        Calculate statistics for the reference distances, including confidence intervals and thresholds.
        """
        ci_lower, ci_upper = scoring.bootstrapping_bca(
            self.reference_distance["metric"],
            self.confidence_level,
            self.statistic,
            self.n_resamples,
            self.random_state,
        )

        estimated_mean = np.mean(self.reference_distance["metric"])
        std_deviation = self.reference_distance["metric"].std()
        lower_threshold = estimated_mean - (3 * std_deviation)
        upper_threshold = estimated_mean + (3 * std_deviation)

        return {
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "mean": estimated_mean,
            "lower_threshold": lower_threshold,
            "upper_threshold": upper_threshold,
        }
