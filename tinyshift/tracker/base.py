# Copyright (c) 2024-2025 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


import numpy as np
from typing import Union, Tuple, List
import pandas as pd
from ..stats import StatisticalInterval
from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(
        self,
        reference: pd.Series,
        drift_limit: Union[str, Tuple[float, float]],
        id_col: str = "unique_id",
    ):
        """
        Initialize the BaseModel class with reference distribution and drift limits.

        Parameters
        ----------
        reference : pd.Series
            Series containing the reference distribution with id_col as grouping variable.
        drift_limit : Union[str, Tuple[float, float]]
            Method for determining drift thresholds ("deviation" or "mad") or custom limits as a tuple.
        id_col : str, default "unique_id"
            Column name used for grouping the reference data.
        """
        self.drift_thresholds_ = reference.groupby(id_col).apply(
            self._get_drift_threshold, drift_limit
        )

    def _get_drift_threshold(
        self,
        reference_metrics: pd.Series,
        drift_limit: Union[str, Tuple[float, float]],
    ) -> float:
        """
        Helper function to compute drift threshold based on specified method or custom limits.
        """
        _, drift_threshold = StatisticalInterval.compute_interval(
            reference_metrics, drift_limit
        )
        return drift_threshold

    def _get_index(self, X: Union[pd.Series, List[np.ndarray], List[list]]):
        """
        Helper function to retrieve the index of a pandas Series or generate a default index.
        """
        return X.index if hasattr(X, "index") else list(range(len(X)))

    def _is_drifted(self, data: pd.Series, group_id: str) -> pd.Series:
        """
        Checks if metrics in the Series are outside the drift threshold for a specific group.
        """
        return data >= self.drift_thresholds_[group_id]

    @abstractmethod
    def score(
        self,
        df: pd.DataFrame,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> pd.Series:
        """
        Compute the drift metric for each time period in the provided dataset.
        """
        pass

    def predict(
        self,
        df: pd.DataFrame,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> pd.Series:
        """
        Predict drift for each time period in the dataset compared to the reference.
        """
        metrics = self.score(
            df,
            id_col,
            time_col,
            target_col,
        )

        metrics["drift"] = metrics.groupby(id_col)["metric"].transform(
            lambda group_data: self._is_drifted(group_data, group_data.name)
        )
        return metrics

    @property
    def thresholds(self) -> list[tuple[float, float]]:
        """Get the drift thresholds for each group."""
        return self.drift_thresholds_
