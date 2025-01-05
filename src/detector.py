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
        thresholds=(),
    ):
        """
        A detector for identifying drift in categorical data over time. The detector uses
        a reference dataset to compute a baseline distribution and compare subsequent data
        for deviations based on a distance metric and statistical thresholds.

        Parameters:
        ----------
        reference_data : DataFrame
            The reference dataset used to compute the baseline distribution.
        target_col : str
            The name of the column containing the categorical variable to analyze.
        datetime_col : str
            The name of the column containing datetime values for temporal grouping.
        period : str
            The frequency for grouping data (e.g., 'D' for daily, 'M' for monthly).
        distance_func : str, optional
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
        thresholds : tuple, optional
            User-defined thresholds for drift detection.
            Default is an empty tuple.

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

        self.period = period
        self.distance_func = distance_func

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
        scoring.check_thresholds(self.statistics, self.reference_distance, thresholds)

    def _calculate_frequency(
        self,
        df,
        target_col,
        datetime_col,
        period,
    ):
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
        Compute a distance metric between consecutive periods in the frequency distribution.

        Parameters:
        ----------
        p : DataFrame
            The frequency distribution with time periods as rows and categorical values as columns.
        distance_func : str
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

        return pd.DataFrame({"datetime": index, "metric": distances[1:]})

    def score(self, analysis, target_col, datetime_col):
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
        freq = self._calculate_frequency(
            analysis, target_col, datetime_col, self.period
        )
        metrics = self._generate_distance(freq, self.distance_func)
        metrics["is_drifted"] = metrics["metric"] >= self.statistics["lower_threshold"]
        return metrics


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
        thresholds=(),
    ):
        """
        A detector for identifying drift in continuous data over time. The detector uses
        a reference dataset to compute a baseline distribution and compare subsequent data
        for deviations using the Kolmogorov-Smirnov test and statistical thresholds.

        Parameters:
        ----------
        reference_data : DataFrame
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
        self.period = period
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
        scoring.check_thresholds(self.statistics, self.reference_ks, thresholds)

    def _calculate_distribution(self, df, column_name, timestamp, period):
        """
        Calculate the continuous distribution of a target column grouped by a given period.

        Parameters:
        ----------
        df : DataFrame
            The dataset to analyze.
        column_name : str
            The name of the column containing the continuous variable.
        timestamp : str
            The name of the datetime column for temporal grouping.
        period : str
            The frequency for grouping (e.g., '1D', '1H').

        Returns:
        -------
        Series
            A Pandas Series containing lists of values grouped by the specified period.
        """
        return (
            df[[timestamp, column_name]]
            .copy()
            .groupby(pd.Grouper(key=timestamp, freq=period))[column_name]
            .agg(list)
        )

    def _generate_ks(self, p):
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

    def score(self, df, target_col, datetime_col):
        """
        Assess drift in the provided dataset by comparing its distribution to the reference.

        Parameters:
        ----------
        df : DataFrame
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
        reference = np.concatenate(self.reference_distribution)
        dist = self._calculate_distribution(df, target_col, datetime_col, self.period)

        metrics = np.array([ks_2samp(reference, row)[1] for row in dist])

        return pd.DataFrame(
            {
                "datetime": dist.index,
                "metric": metrics,
                "is_drifted": metrics <= self.statistics["lower_threshold"],
            },
        )
