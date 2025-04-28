import numpy as np
from sklearn.metrics import f1_score
import pandas as pd
from .base import BaseModel
from typing import Callable, Tuple, Union, List


class PerformanceTracker(BaseModel):
    def __init__(
        self,
        y: Union[pd.Series, List[np.ndarray], List[list]],
        y_pred: Union[pd.Series, List[np.ndarray], List[list]],
        metric_score: Callable = f1_score,
        statistic: Callable = np.mean,
        confidence_level: float = 0.997,
        n_resamples: int = 1000,
        random_state: int = 42,
        drift_limit: Union[str, Tuple[float, float]] = "stddev",
        confidence_interval: bool = False,
    ):
        """
        Initialize a tracker for monitoring model performance over time using a specified evaluation metric.
        The tracker compares the performance metric across time periods to a reference distribution
        and identifies potential performance degradation.

        Parameters
        ----------
        y : Union[pd.Series, List[np.ndarray], List[list]]
            The actual target values. Can be a pandas Series or a list of lists/arrays.
        y_pred : Union[pd.Series, List[np.ndarray], List[list]]
            The predicted values. Can be a pandas Series or a list of lists/arrays.
        metric_score : Callable, optional
            The function to compute the evaluation metric (e.g., `f1_score`).
            Default is `f1_score`.
        statistic : Callable, optional
            The statistic function used to summarize the reference metric distribution.
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
            The method or thresholds for drift detection. If "stddev", thresholds are based on standard deviation.
            If a tuple, it specifies custom lower and upper thresholds.
            Default is "stddev".
        confidence_interval : bool, optional
            Whether to calculate confidence intervals for the metric distribution.
            Default is False.

        Attributes
        ----------
        metric_score : Callable
            The evaluation metric function used for tracking performance.
        reference_distribution : pd.Series
            The performance metric distribution of the reference dataset.

        Raises
        ------
        TypeError
            If `metric_score` is not a callable function.
        """

        if not callable(metric_score):
            raise TypeError("metric_score must be a callable function.")

        self.metric_score = metric_score

        self.reference_distribution = self.score(y, y_pred)
        super().__init__(
            self.reference_distribution,
            confidence_level,
            statistic,
            n_resamples,
            random_state,
            drift_limit,
            confidence_interval,
        )

    def score(
        self,
        y: Union[pd.Series, List[np.ndarray], List[list]],
        y_pred: Union[pd.Series, List[np.ndarray], List[list]],
    ):
        """
        Parameters
        y : Union[pd.Series, List[np.ndarray], List[list]]
            The actual target values. Can be a pandas Series or a list of arrays/lists.
        y_pred : Union[pd.Series, List[np.ndarray], List[list]]
            The predicted values. Can be a pandas Series or a list of arrays/lists.

        Returns
            A pandas Series containing the computed metric for each pair of inputs.

        Raises
        ------
        ValueError
            If `y` and `y_pred` do not have the same length.
        """
        if len(y) != len(y_pred):
            raise ValueError("y and y_pred must have the same length.")

        return pd.Series(
            [
                self.metric_score(target, prediction)
                for target, prediction in zip(y, y_pred)
            ],
            index=y.index if isinstance(y, pd.Series) else range(len(y)),
        )

    def predict(self, y, y_pred) -> pd.DataFrame:
        """Predict drift for each time period in the dataset compared to the reference."""
        metrics = self.score(y, y_pred)
        return self._is_drifted(metrics)
