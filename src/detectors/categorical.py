import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from ..base.model import BaseModel
from typing import Callable, Tuple, Union


def l_infinity(a, b):
    """
    Compute the L-infinity distance between two distributions.
    """
    return np.max(np.abs(a - b))


class CategoricalDriftDetector(BaseModel):
    def __init__(
        self,
        reference: pd.DataFrame,
        target_col: str,
        datetime_col: str,
        period: str,
        func: str = "l_infinity",
        statistic: Callable = np.mean,
        confidence_level: float = 0.997,
        n_resamples: int = 1000,
        random_state: int = 42,
        drift_limit: Union[str, Tuple[float, float]] = "deviation",
    ):
        """
        A detector for identifying drift in categorical data over time. The detector uses
        a reference dataset to compute a baseline distribution and compare subsequent data
        for deviations based on a distance metric and drift limits.

        Parameters:
        ----------
        reference : DataFrame
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
        statistic : callable, optional
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
        drift_limit : tuple, optional
            User-defined thresholds for drift detection.
            Default is the 'deviation method'.

        Attributes:
        ----------
        period : str
            The grouping frequency used for analysis.
        reference_frequency : DataFrame
            The frequency distribution of the reference dataset.
        reference_distance : DataFrame
            The distance metric values for the reference dataset.
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
        self.func = func

        # Initialize frequency and statistics
        self.reference_distribution = self._calculate_frequency(
            reference,
            target_col,
            datetime_col,
            period,
        )

        self.reference_distance = self._generate_distance(
            self.reference_distribution,
            func,
        )

        super().__init__(
            self.reference_distance,
            confidence_level,
            statistic,
            n_resamples,
            random_state,
            drift_limit,
        )

    def _calculate_frequency(
        self,
        df: pd.DataFrame,
        target_col: str,
        datetime_col: str,
        period: str,
    ) -> pd.DataFrame:
        """
        Calculate the frequency distribution of the target column grouped by a time period.

        Parameters:
        ----------
        df : DataFrame
            The dataset to analyze.
        target_col : str
            The name of the categorical column.
        datetime_col : str
            The name of the datetime column for temporal grouping.
        period : str
            The frequency for grouping (e.g., 'D', 'M').

        Returns:
        -------
        DataFrame
            A pivot table of frequencies with time periods as rows and categorical
            values as columns.
        """
        freq = (
            df.groupby([pd.Grouper(key=datetime_col, freq=period), target_col])
            .size()
            .unstack(fill_value=0)
        )

        return freq

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
        func_name: str,
    ) -> pd.DataFrame:
        """
        Compute a distance metric between consecutive periods in the frequency distribution.

        Parameters:
        ----------
        p : DataFrame
            The frequency distribution with time periods as rows and categorical values as columns.
        func : str
            The distance function to use ('l_infinity' or 'jensenshannon').

        Returns:
        -------
        DataFrame
            A DataFrame containing datetime values and the calculated distances.
        """
        n = p.shape[0]
        distances = np.zeros(n)
        past_value = np.zeros(p.shape[1], dtype=np.int32)
        index = p.index[1:]
        p = np.asarray(p)
        func = self._selection_function(func_name)

        for i in range(1, n):
            past_value = past_value + p[i - 1]
            past_value = past_value / np.sum(past_value)
            current_value = p[i] / np.sum(p[i])
            dist = func(past_value, current_value)
            distances[i] = dist

        return pd.DataFrame({"datetime": index, "metric": distances[1:]})

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
            The name of the categorical column in the analysis dataset.
        datetime_col : str
            The name of the datetime column in the analysis dataset.

        Returns:
        -------
        DataFrame
            A DataFrame containing metrics and drift detection results for each time period.
        """
        if target_col not in analysis.columns:
            raise KeyError(f"Column {target_col} is not in the DataFrame.")
        if datetime_col not in analysis.columns:
            raise KeyError(f"Datetime column {datetime_col} is not in the DataFrame.")
        if not pd.api.types.is_datetime64_any_dtype(analysis[datetime_col]):
            raise TypeError(f"Column {datetime_col} must be of datetime type.")

        freq = self._calculate_frequency(
            analysis, target_col, datetime_col, self.period
        )
        metrics = self._generate_distance(freq, self.func)
        metrics["is_drifted"] = self.is_drifted(metrics)
        return metrics
