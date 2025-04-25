import numpy as np
import pandas as pd
from .base import BaseModel
from typing import Callable, Tuple, Union
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
        A tracker for monitoring anomaly over time using a specified evaluation metric.
        The tracker compares the performance metric across time periods to a reference distribution
        and identifies potential performance degradation.

        Parameters:
        ----------
        anomaly_model : Union[SPAD, HBOS]
            The anomaly detection model used to calculate anomaly scores.
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
            User-defined thresholds for drift detection.
            Default is "stddev".

        Attributes:
        ----------
        anomaly_model : Union[SPAD, HBOS]
            The anomaly detection model instance.
        anomaly_scores : DataFrame
            DataFrame containing the anomaly scores.
        """
        self.anomaly_model = anomaly_model
        if not 0 < confidence_level <= 1:
            raise ValueError("confidence_level must be between 0 and 1.")
        if n_resamples <= 0:
            raise ValueError("n_resamples must be a positive integer.")

        self.anomaly_scores = pd.DataFrame(
            self.anomaly_model.decision_scores_, columns=["metric"]
        )
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
        analysis: pd.DataFrame,
    ):
        """
        Calculate the anomaly scores for the given dataset and determine if there is a drift.

        Parameters:
        ----------
        analysis : DataFrame
            The dataset to analyze for anomalies.

        Returns:
        -------
        DataFrame
            A DataFrame containing the calculated anomaly scores and a boolean
            indicating whether drift was detected for each record.
        """

        metrics = pd.DataFrame()
        metrics["metric"] = self.anomaly_model.decision_function(analysis)
        metrics["is_drifted"] = self._is_drifted(metrics)
        return metrics
