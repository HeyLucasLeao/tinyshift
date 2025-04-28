import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from .base import BaseModel
from typing import Callable, Tuple, Union
from collections import Counter


def l_infinity(a, b):
    """
    Compute the L-infinity distance between two distributions.
    """
    return np.max(np.abs(a - b))


class CategoricalDriftTracker(BaseModel):
    def __init__(
        self,
        reference: pd.DataFrame,
        func: str = "l_infinity",
        statistic: Callable = np.mean,
        confidence_level: float = 0.997,
        n_resamples: int = 1000,
        random_state: int = 42,
        drift_limit: Union[str, Tuple[float, float]] = "stddev",
        confidence_interval: bool = False,
    ):
        """
        A tracker for identifying drift in categorical data over time. The tracker uses
        a reference dataset to compute a baseline distribution and compare subsequent data
        for deviations based on a distance metric and drift limits.

        Parameters:
        ----------
        reference : pd.DataFrame
            The reference dataset used to compute the baseline distribution.
        target_col : str
            The name of the column containing the categorical variable to analyze.
        datetime_col : str
            The name of the column containing datetime values for temporal grouping.
        period : str
            The frequency for grouping data (e.g., 'D' for daily, 'M' for monthly).
        func : str, optional
            The distance function to use ('l_infinity' or 'jensenshannon').
            Default is 'l_infinity'.
        statistic : Callable, optional
            The statistic function used to summarize the reference distances.
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
        drift_limit : Union[str, Tuple[float, float]], optional
            User-defined thresholds for drift detection. If set to "stddev", thresholds
            are calculated based on the standard deviation of the reference distances.
            Default is "stddev".
        confidence_interval : bool, optional
            Whether to calculate and include confidence intervals in the drift analysis.
            Default is False.

        Attributes:
        ----------
        func : Callable
            The distance function used for drift calculation.
        reference_frequency : pd.DataFrame
            The frequency distribution of the reference dataset.
        reference_distance : pd.DataFrame
            The distance metric values for the reference dataset.
        """

        self.func = self._selection_function(func)

        reference_frequency = self._calculate_frequency(
            reference,
        )

        self.reference_distribution = reference_frequency.sum(axis=0) / np.sum(
            reference_frequency.sum(axis=0)
        )

        self.reference_distance = self._generate_distance(
            reference_frequency,
        )

        super().__init__(
            self.reference_distance,
            confidence_level,
            statistic,
            n_resamples,
            random_state,
            drift_limit,
            confidence_interval,
        )

    def _calculate_frequency(
        self,
        data: np.ndarray,
    ) -> pd.DataFrame:
        """
        Calculates the percent distribution of a categorical column grouped by a specified time period.
        """
        index = data.index if isinstance(data, pd.Series) else list(range(len(data)))
        data = np.asanyarray(data)
        freq = [Counter(item) for item in data]
        categories = np.unique(np.concatenate(data))
        return pd.DataFrame(freq, columns=categories, index=index)

    def _selection_function(self, func_name: str) -> Callable:
        """Returns a specific function based on the given function name."""

        if func_name == "l_infinity":
            selected_func = l_infinity
        elif func_name == "jensenshannon":
            selected_func = jensenshannon
        else:
            raise ValueError(f"Unsupported distance function: {func_name}")
        return selected_func

    def _generate_distance(
        self,
        p: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        This method calculates a distance metric between consecutive rows of a frequency
        distribution DataFrame, where rows represent time periods and columns represent
        categorical values. The distance is computed using a specified function.

        Parameters
        p : pd.DataFrame
            A DataFrame representing the frequency distribution, where rows correspond to
            time periods and columns correspond to categorical values.

        Returns
        pd.DataFrame
            A DataFrame containing two columns:
            - 'datetime': The datetime values corresponding to the time periods (excluding the first period).
            - 'metric': The calculated distance metric for each consecutive period.
        """
        n = p.shape[0]
        distances = np.zeros(n)
        past_value = np.zeros(p.shape[1], dtype=np.int32)
        index = p.index[1:]
        p = np.asarray(p)

        for i in range(1, n):
            past_value = past_value + p[i - 1]
            past_value = past_value / np.sum(past_value)
            current_value = p[i] / np.sum(p[i])
            dist = self.func(past_value, current_value)
            distances[i] = dist

        return pd.DataFrame({"datetime": index, "metric": distances[1:]})

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
            The name of the categorical column in the analysis dataset.
        datetime_col : str
            The name of the datetime column in the analysis dataset.

        Returns:
        -------
        DataFrame
            A DataFrame containing metrics and drift detection results for each time period.
        """

        # Calculate frequency and percentage distribution
        freq = self._calculate_frequency(analysis)
        percent = freq.div(freq.sum(axis=1), axis=0)

        # Calculate drift metrics for each time period
        percent["metric"] = percent.apply(
            lambda row: self.func(row, self.reference_distribution), axis=1
        )

        percent["is_drifted"] = self._is_drifted(percent)

        return percent
