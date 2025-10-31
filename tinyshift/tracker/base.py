# Copyright (c) 2024-2025 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


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
        drift_limit: Union[str, Tuple[float, float]],
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
        drift_limit : Union[str, Tuple[float, float]]
            Method for determining drift thresholds ("deviation" or "mad") or custom limits as a tuple.
        """

        self.lower_bounds_, self.upper_bounds_ = StatisticalInterval.compute_interval(
            reference, drift_limit
        )

    def _get_index(self, X: Union[pd.Series, List[np.ndarray], List[list]]):
        """
        Helper function to retrieve the index of a pandas Series or generate a default index.
        """
        return X.index if hasattr(X, "index") else list(range(len(X)))

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

        lower_limit = self.lower_bounds_
        upper_limit = self.upper_bounds_

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

    @property
    def bounds_(self) -> list[tuple[float, float]]:
        """Get the winsorization bounds for each feature as list of (lower, upper) tuples."""
        return list(zip(self.lower_bounds_, self.upper_bounds_))
