from ..plot import plot
import numpy as np
from scipy.stats import norm
from typing import Callable, Union, Tuple, Dict
import pandas as pd


class BaseModel:
    def __init__(
        self,
        reference: pd.DataFrame,
        confidence_level: float,
        statistic: Callable,
        n_resamples: int,
        random_state: int,
        drift_limit: Union[str, Tuple[float, float]],
    ):
        """
        Initializes the BaseModel class with reference distribution, statistics, and drift limits.

        Parameters:
        ----------
        reference : pd.DataFrame
            Data containing the reference distribution with a "metric" column.
        confidence_level : float
            Desired confidence level for statistical calculations (e.g., 0.95).
        statistic : Callable
            Function to compute summary statistics (e.g., np.mean).
        n_resamples : int
            Number of bootstrap resamples.
        random_state : int
            Seed for reproducibility.
        drift_limit : Union[str, Tuple[float, float]]
            Method ("deviation" or "mad") or custom limits for drift thresholding.
        """

        self.statistics = self._statistic_generate(
            reference,
            confidence_level,
            statistic,
            n_resamples,
            random_state,
        )
        self.plot = plot.Plot(self.statistics, reference)
        self._drift_limit_generate(self.statistics, reference, drift_limit)

    def _jackknife_acceleration(self, data: np.ndarray, statistic: Callable) -> float:
        """Calculate the acceleration parameter using jackknife resampling."""
        n = len(data)
        jackknife = np.array([statistic(np.delete(data, i)) for i in range(n)])
        jackknife_mean = jackknife.mean()
        diffs = jackknife - jackknife_mean
        acceleration = np.sum(diffs**3) / (6.0 * (np.sum(diffs**2) ** 1.5))
        return acceleration

    def _bootstrap_statistics(
        self, data: np.ndarray, statistic: Callable, n_resamples: int
    ) -> np.ndarray:
        """Perform bootstrap resampling and calculate statistics."""
        return np.array(
            [
                statistic(np.random.choice(data, size=len(data), replace=True))
                for _ in range(n_resamples)
            ]
        )

    def _bootstrapping_bca(
        self,
        data: pd.Series,
        confidence_level: float,
        statistic: Callable,
        n_resamples: int,
        random_state: int,
    ) -> Union[float, float]:
        """
        Calculates the bias-corrected and accelerated (BCa) bootstrap confidence interval for the given data.

        Parameters:
        - data (list or numpy array): Sample data.
        - confidence_level (float): Desired confidence level (e.g., 0.95 for 95%).
        - statistic (function): Statistical function to apply to the data. Default is np.mean.
        - n_resamples (int): Number of bootstrap resamples to perform. Default is 1000.
        - random_state (int): Random seed for reproducibility.

        Returns:
        - tuple: A tuple containing the lower and upper bounds of the BCa confidence interval.
        """
        np.random.seed(random_state)
        data = np.asarray(data)

        # Bootstrap resampling
        sample_statistics = self._bootstrap_statistics(data, statistic, n_resamples)

        # Jackknife resampling for acceleration
        acceleration = self._jackknife_acceleration(data, statistic)

        # Bias correction
        observed_stat = statistic(data)
        bias = np.mean(sample_statistics < observed_stat)
        z0 = norm.ppf(bias)

        # Adjusting percentiles
        alpha = 1 - confidence_level
        z_alpha = norm.ppf(1 - alpha / 2)

        z_lower_bound = (z0 - z_alpha) / (1 - acceleration * (z0 - z_alpha)) + z0
        z_upper_bound = (z0 + z_alpha) / (1 - acceleration * (z0 + z_alpha)) + z0

        alpha_lower = norm.cdf(z_lower_bound)
        alpha_upper = norm.cdf(z_upper_bound)

        # Calculate lower and upper bounds from the percentiles
        lower_bound = np.quantile(sample_statistics, alpha_lower)
        upper_bound = np.quantile(sample_statistics, alpha_upper)

        return lower_bound, upper_bound

    def _statistic_generate(
        self,
        df: pd.DataFrame,
        confidence_level: float,
        statistic: Callable,
        n_resamples: int,
        random_state: int,
    ):
        """
        Calculate statistics for the reference distances, including confidence intervals and thresholds.
        """
        ci_lower, ci_upper = self._bootstrapping_bca(
            df["metric"],
            confidence_level,
            statistic,
            n_resamples,
            random_state,
        )
        estimated_mean = np.mean(df["metric"])

        return {
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "mean": estimated_mean,
        }

    def _custom_threshold(
        self, data: pd.Series, custom_func: Callable
    ) -> Tuple[float, float]:
        """Calculate thresholds using a custom function."""
        return custom_func(data)

    def _calculate_threshold(
        self, data: pd.Series, center: Callable, spread: Callable, factor: float = 3
    ) -> Tuple[float, float]:
        """Calculate thresholds using a central tendency and spread function."""
        center_value = center(data)
        spread_value = spread(data)
        lower_limit = center_value - factor * spread_value
        upper_limit = center_value + factor * spread_value
        return lower_limit, upper_limit

    def _iqr_threshold(self, df: pd.DataFrame) -> Tuple[float, float]:
        """
        Calculates thresholds using IQR and median with a default factor of 1.5.
        """

        def iqr(x):
            q75, q25 = np.percentile(x, [75, 25])
            return q75 - q25

        return self._calculate_threshold(df["metric"], np.median, iqr, factor=1.5)

    def _stddev_threshold(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Calculates thresholds using mean and standard deviation."""
        return self._calculate_threshold(df["metric"], np.mean, np.std)

    def _mad_threshold(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Calculates thresholds using Median Absolute Deviation (MAD)."""
        mad = lambda x: np.median(np.abs(x - np.median(x)))
        return self._calculate_threshold(df["metric"], np.median, mad)

    def _drift_limit_generate(
        self,
        statistics: Dict,
        distribution: pd.DataFrame,
        drift_limit: Union[str, Callable, Tuple[float]],
    ):
        """
        Determines the drift limits based on the specified method.
        """
        if isinstance(drift_limit, str):
            if drift_limit == "stddev":
                lower_limit, upper_limit = self._stddev_threshold(distribution)
            elif drift_limit == "mad":
                lower_limit, upper_limit = self._mad_threshold(distribution)
            elif drift_limit == "iqr":
                lower_limit, upper_limit = self._iqr_threshold(distribution)
            else:
                raise ValueError(f"Unsupported drift limit method: {drift_limit}")
        elif callable(drift_limit):
            lower_limit, upper_limit = self._custom_threshold(
                distribution["metric"], drift_limit
            )
        elif isinstance(drift_limit, tuple) and len(drift_limit) == 2:
            lower_limit, upper_limit = drift_limit
        else:
            raise ValueError("Invalid drift limit specification.")

        # Update the statistics dictionary with the new thresholds
        statistics["lower_limit"] = lower_limit
        statistics["upper_limit"] = upper_limit

    def _validate_columns(
        self,
        df: pd.DataFrame,
        target_col: str,
        datetime_col: str,
    ):
        if target_col not in df.columns:
            raise KeyError(f"Column {target_col} is not in the DataFrame.")
        if datetime_col not in df.columns:
            raise KeyError(f"Datetime column {datetime_col} is not in the DataFrame.")
        if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
            raise TypeError(f"Column {datetime_col} must be of datetime type.")

    def _validate_params(
        self,
        confidence_level: float,
        n_resamples: int,
        period: str,
    ):
        if not 0 < confidence_level <= 1:
            raise ValueError("confidence_level must be between 0 and 1.")
        if n_resamples <= 0:
            raise ValueError("n_resamples must be a positive integer.")
        if not isinstance(period, str):
            raise TypeError("period must be a string (e.g., 'W', 'M').")

    def _is_drifted(self, df: pd.DataFrame) -> pd.Series:
        """
        Checks if metrics in the DataFrame are outside specified limits
        and returns the drift status.
        """
        is_drifted = pd.Series([False] * len(df))

        if self.statistics["lower_limit"] is not None:
            is_drifted |= df["metric"] <= self.statistics["lower_limit"]
        if self.statistics["upper_limit"] is not None:
            is_drifted |= df["metric"] >= self.statistics["upper_limit"]

        return is_drifted
