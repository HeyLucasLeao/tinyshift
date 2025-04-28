import numpy as np
import pandas as pd
from .base import BaseModel
from typing import Callable, Tuple, Union, List
from ..outlier import *


class AnomalyTracker(BaseModel):
    def __init__(
        self,
        anomaly_model: Union[SPAD, HBOS],
        statistic: Callable = np.mean,
        confidence_level: float = 0.997,
        n_resamples: int = 1000,
        random_state: int = 42,
        drift_limit: Union[str, Tuple[float, float]] = "stddev",
        confidence_interval: bool = False,
    ):
        """
        A tracker for monitoring anomalies over time using a specified evaluation metric.
        The tracker compares the performance metric across time periods to a reference distribution
        and identifies potential performance degradation.

        Parameters
        ----------
        anomaly_model : Union[SPAD, HBOS]
            The anomaly detection model used to calculate anomaly scores.
        statistic : Callable, optional
            The statistic function used to summarize the reference metric distribution.
            Default is `np.mean`.
        confidence_level : float, optional
            The confidence level for calculating statistical thresholds.
            Must be between 0 and 1. Default is 0.997.
        n_resamples : int, optional
            Number of resamples for bootstrapping when calculating statistics.
            Must be a positive integer. Default is 1000.
        random_state : int, optional
            Seed for reproducibility of random resampling.
            Default is 42.
        drift_limit : Union[str, Tuple[float, float]], optional
            User-defined thresholds for drift detection. Can be "stddev" or a tuple of floats.
            Default is "stddev".
        confidence_interval : bool, optional
            Whether to calculate and include confidence intervals in the analysis.
            Default is False.

        Attributes
        ----------
        anomaly_model : Union[SPAD, HBOS]
            The anomaly detection model instance.
        anomaly_scores : DataFrame
            DataFrame containing the anomaly scores.
        """
        self.anomaly_model = anomaly_model
        self.anomaly_scores = self.anomaly_model.decision_scores_

        super().__init__(
            self.anomaly_scores,
            confidence_level,
            statistic,
            n_resamples,
            random_state,
            drift_limit,
            confidence_interval,
        )

    def score(
        self,
        X: Union[pd.Series, np.ndarray, List[float], List[int]],
    ):
        """
        Calculate the anomaly scores for the given dataset.

        This method uses the anomaly detection model to compute scores for the input data,
        which can be used to identify potential anomalies.

        Parameters
        X : Union[pd.Series, np.ndarray, List[float], List[int]]
            The input data for which anomaly scores are to be calculated.

        Returns
        np.ndarray
            An array containing the calculated anomaly scores for each record in the input data.
        """
        scores = self.anomaly_model.decision_function(X)
        return pd.Series(scores, index=self._get_index(X))
