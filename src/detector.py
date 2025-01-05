import numpy as np
import pandas as pd
from src import scoring
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp
from . import plot


def l_infinity(a, b):
    """
    Compute the L-infinity distance between two distributions.
    """
    return np.max(np.abs(a - b))


class CategoricalDriftDetector:
    def __init__(
        self,
        reference_data,
        target_col,
        datetime_col,
        period,
        distance_func="l_infinity",
        statistic=np.mean,
        confidence_level=0.997,
        n_resamples=1000,
        random_state=42,
    ):

        # Initialize frequency and statistics
        self.reference_frequency = self._calculate_frequency(
            reference_data,
            target_col,
            datetime_col,
            period,
        )

        self.reference_distance = self._generate_distance(
            self.reference_frequency,
            distance_func,
        )

        self.statistics = scoring.calculate_statistics(
            self.reference_distance,
            confidence_level,
            statistic,
            n_resamples=n_resamples,
            random_state=random_state,
        )
        self.plot = plot.Plot(self.statistics, self.reference_distance)

    def _calculate_frequency(
        self,
        df,
        target_col,
        datetime_col,
        period,
    ):
        """
        Calculate the reference frequency by grouping data based on the specified period.
        """
        grouped = (
            df.groupby(pd.Grouper(key=datetime_col, freq=period))
            .apply(lambda x: x[target_col].value_counts())
            .rename("metric")
            .reset_index()
            .rename(columns={"level_1": target_col})
        )

        grouped = grouped.set_index([datetime_col, target_col]).unstack(fill_value=0)
        return grouped

    def _generate_distance(self, p, distance_func):
        """
        Generate the distance metric based on the specified distance function.
        """
        n = p.shape[0]
        distances = np.zeros(n)
        past_value = np.zeros(p.shape[1], dtype=np.int32)
        index = p.index[1:]
        p = np.asarray(p)

        if distance_func == "l_infinity":
            func = l_infinity
        elif distance_func == "jensenshannon":
            func = jensenshannon
        else:
            raise ValueError(f"Unsupported distance function: {distance_func}")

        for i in range(1, n):
            past_value = past_value + p[i - 1]
            past_value = past_value / np.sum(past_value)
            current_value = p[i] / np.sum(p[i])
            dist = func(past_value, current_value)
            distances[i] = dist

        return pd.DataFrame({"datetime": index, "metric": distances[1:]}).reset_index()


class ContinuousDriftDetector:
    def __init__(
        self,
        reference_data,
        target_col,
        datetime_col,
        period,
        statistic=np.mean,
        confidence_level=0.997,
        n_resamples=1000,
        random_state=42,
    ):

        # Initialize frequency and statistics
        self.reference_distribution = self._calculate_distribution(
            reference_data,
            target_col,
            datetime_col,
            period,
        )

        self.reference_ks = self._generate_ks(
            self.reference_distribution,
        )

        self.statistics = scoring.calculate_statistics(
            self.reference_ks,
            confidence_level,
            statistic,
            n_resamples=n_resamples,
            random_state=random_state,
        )
        self.plot = plot.Plot(self.statistics, self.reference_ks)

    def _calculate_distribution(self, df, column_name, timestamp, period):
        """
        Calculates the continuous distribution grouped by a given period.

        Args:
            df (pd.DataFrame): DataFrame containing the data.
            column_name (str): Name of the column to aggregate.
            period (str): Period for grouping (e.g., '1D', '1H').
            timestamp (str): Name of the timestamp column.

        Returns:
            pd.Series: Series containing lists of values grouped by the specified period.
        """
        return (
            df[[timestamp, column_name]]
            .copy()
            .groupby(pd.Grouper(key=timestamp, freq=period))[column_name]
            .agg(list)
        )

    def _generate_ks(self, p):
        """
        Calculates a metric based on the Kolmogorov-Smirnov test in a cummulative rolling window.

        Args:
            p (pd.Series): Series of lists representing distributions for each period.

        Returns:
            pd.DataFrame: DataFrame with the indices and calculated metric.
        """
        n = p.shape[0]
        p_values = np.zeros(n)
        past_values = np.array([], dtype=float)

        for i in range(1, n):
            past_values = np.concatenate([past_values, p[i - 1]])
            _, p_value = ks_2samp(past_values, p[i])
            p_values[i] = p_value

        return pd.DataFrame({"datetime": p.index[1:], "metric": p_values[1:]})
