import numpy as np
import pandas as pd
from sklearn.utils import check_array
from collections import Counter
from .base import BaseHistogramModel
from typing import Union


class HBOS(BaseHistogramModel):
    """
    HBOS (Histogram-based Outlier Score) is an unsupervised outlier detection algorithm that
    uses histograms to model the distribution of features and compute outlier scores.

    References
    ----------
    Goldstein, Markus & Dengel, Andreas. (2012). Histogram-based Outlier Score (HBOS): A fast Unsupervised Anomaly Detection Algorithm.
    https://www.researchgate.net/publication/231614824_Histogram-based_Outlier_Score_HBOS_A_fast_Unsupervised_Anomaly_Detection_Algorithm
    """

    def __init__(self):
        super().__init__()

    def fit(
        self,
        X: np.ndarray,
        nbins: Union[int, str] = 10,
        dynamic_bins: bool = False,
    ) -> "HBOS":
        """
        Fit the HBOS model according to the given training data.
        Parameters
        ----------
        X : np.ndarray
            Training data, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        nbins : Union[int, str], optional (default=10)
            Number of bins to use for the histogram. If 'auto', the number of bins
            is determined automatically.
        dynamic_bins : bool, optional (default=False)
            If True, the number of bins is determined dynamically for each feature.
        Returns
        -------
        self : HBOS
            Fitted estimator.
         Notes
        -----
        - If `X` is a pandas Series or DataFrame, the data types and column names are stored.
        - For categorical features, relative frequencies are computed.
        - For continuous features, the data is discretized into bins and densities are computed.
        - The decision scores are computed and stored in `self.decision_scores_`.
        """
        self._extract_feature_info(X)

        X = check_array(X)
        _, self.n_features = X.shape

        for i in range(self.n_features):
            nbins = self._check_bins(X[:, i], nbins)

            if isinstance(self.feature_dtypes[i], pd.CategoricalDtype):
                value_counts = Counter(X[:, i])
                total_values = sum(value_counts.values())
                relative_frequencies = {
                    value: (count + 1) / (total_values + len(value_counts))
                    for value, count in value_counts.items()
                }
                self.feature_distributions.append(relative_frequencies)
            elif dynamic_bins:
                percentiles = np.percentile(X[:, i], q=np.linspace(0, 100, nbins + 1))
                bin_edges = np.unique(percentiles)
                densities, _ = np.histogram(X[:, i], bins=bin_edges, density=True)
                self.feature_distributions.append([densities, bin_edges])
            else:
                densities, bin_edges = np.histogram(X[:, i], bins=nbins, density=True)
                self.feature_distributions.append([densities, bin_edges])

        self.decision_scores_ = self.decision_function(X)
        return self

    def _compute_outlier_score(self, X: np.ndarray, i: int) -> np.ndarray:
        """
        Compute the outlier score for a specific feature in the dataset.
        """

        if isinstance(self.feature_dtypes[i], pd.CategoricalDtype):
            density = np.array(
                [self.feature_distributions[i].get(value, 1e-9) for value in X[:, i]]
            )
        else:
            densities, bin_edges = self.feature_distributions[i]
            digitized = np.digitize(X[:, i], bin_edges, right=True)
            bin_indices = np.clip(digitized - 1, 0, len(densities) - 1)
            density = densities[bin_indices]
        return -np.log(density + 1e-9)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the decision function for the input data.
        """

        self._check_columns(X)

        X = check_array(X)
        outlier_scores = np.zeros(shape=(X.shape[0], self.n_features))

        for i in range(self.n_features):
            outlier_scores[:, i] = self._compute_outlier_score(X, i)

        return np.sum(outlier_scores, axis=1).ravel() * -1
