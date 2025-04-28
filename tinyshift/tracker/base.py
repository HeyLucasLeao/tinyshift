from ..plot import plot
import numpy as np
from typing import Callable, Union, Tuple, List
import pandas as pd
from ..stats import StatisticalInterval, BootstrapBCA
from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(
        self,
        reference: pd.Series,
        confidence_level: float,
        statistic: Callable,
        n_resamples: int,
        random_state: int,
        drift_limit: Union[str, Tuple[float, float]],
        confidence_interval: bool,
    ):
        """
        Initialize the BaseModel class with reference distribution, statistics, and drift limits.

        Parameters
        ----------
        reference : pd.Series
            Series containing the reference distribution.
        confidence_level : float
            Confidence level for statistical calculations (e.g., 0.95).
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

        if not 0 < confidence_level <= 1:
            raise ValueError("confidence_level must be between 0 and 1.")
        if n_resamples <= 0:
            raise ValueError("n_resamples must be a positive integer.")

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
            StatisticalInterval.compute_interval(reference, drift_limit)
        )

    def _generate_statistics(
        self,
        data: pd.Series,
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
                data,
                confidence_level,
                statistic,
                n_resamples,
                random_state,
            )

        return {
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "mean": np.mean(data),
        }

    def _get_index(self, X: Union[pd.Series, List[np.ndarray], List[list]]):
        """
        Helper function to retrieve the index of a pandas Series or generate a default index.
        """
        return X.index if isinstance(X, pd.Series) else list(range(len(X)))

    def _is_drifted(self, data: pd.Series) -> pd.Series:
        """
        Checks if metrics in the Series are outside specified limits
        and returns the drift status as a boolean Series.

        Parameters
        ----------
        data : pd.Series
            A Series containing the metrics to be checked against the drift limits.

        Returns
        -------
        pd.Series
            A boolean Series indicating whether each metric is drifted (True) or not (False).
        """
        is_drifted = pd.Series(False, index=data.index, dtype=bool)

        lower_limit = self.statistics.get("lower_limit")
        upper_limit = self.statistics.get("upper_limit")

        if lower_limit is not None:
            is_drifted |= data <= lower_limit
        if upper_limit is not None:
            is_drifted |= data >= upper_limit

        return is_drifted

    @abstractmethod
    def score(
        self,
        X: Union[pd.Series, List[np.ndarray], List[list]],
    ) -> pd.Series:
        """
        Compute the drift metric for each time period in the provided dataset.
        """
        pass

    def predict(self, X: Union[pd.Series, List[np.ndarray], List[list]]) -> pd.Series:
        """Predict drift for each time period in the dataset compared to the reference."""
        metrics = self.score(X)
        return self._is_drifted(metrics)
