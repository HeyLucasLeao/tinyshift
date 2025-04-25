from ..plot import plot
import numpy as np
from typing import Callable, Union, Tuple
import pandas as pd
from ..stats import StatisticalInterval, BootstrapBCA


class BaseModel:
    def __init__(
        self,
        reference: pd.DataFrame,
        confidence_level: float,
        statistic: Callable,
        n_resamples: int,
        random_state: int,
        drift_limit: Union[str, Tuple[float, float]],
        confidence_interval: bool,
    ):
        """
        Initializes the BaseModel class with reference distribution, statistics, and drift limits.

        Parameters
        ----------
        reference : pd.DataFrame
            Data containing the reference distribution with a "metric" column.
        confidence_level : float
            Desired confidence level for statistical calculations (e.g., 0.95).
        statistic : Callable
            Function to compute summary statistics (e.g., np.mean).
        n_resamples : int
            Number of bootstrap resamples for confidence interval estimation.
        random_state : int
            Seed for reproducibility of bootstrap resampling.
        drift_limit : Union[str, Tuple[float, float]]
            Method for determining drift thresholds ("deviation" or "mad") or custom limits as a tuple.
        confidence_interval : bool
            Whether to compute confidence intervals for the reference distribution.
        """

        self.confidence_interval = confidence_interval
        self.statistics = self._generate_statistics(
            reference,
            confidence_level,
            statistic,
            n_resamples,
            random_state,
        )
        self.plot = plot.Plot(self.statistics, reference, self.confidence_interval)

        self.statistics["lower_limit"], self.statistics["upper_limit"] = (
            StatisticalInterval.compute_interval(reference["metric"], drift_limit)
        )

    def _generate_statistics(
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
        ci_lower, ci_upper = (None, None)

        if self.confidence_interval:
            ci_lower, ci_upper = BootstrapBCA.compute_interval(
                df["metric"],
                confidence_level,
                statistic,
                n_resamples,
                random_state,
            )

        return {
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "mean": np.mean(df["metric"]),
        }

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
