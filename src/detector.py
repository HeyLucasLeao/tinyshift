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
