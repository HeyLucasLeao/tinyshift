import numpy as np
import pandas as pd
from sklearn.utils import check_array
from collections import Counter
from sklearn.decomposition import PCA
from .base import BaseHistogramModel
from typing import Union
from ..stats import StatisticalInterval


class SPAD(BaseHistogramModel):
    """
    SPAD (Statistical Probability Anomaly Detection) detects outliers by discretizing continuous data into bins and calculating anomaly scores based on the logarithm of inverse probabilities for each feature.

    SPAD+ enhances SPAD by incorporating Principal Components (PCs) from PCA, capturing feature correlations to detect multivariate anomalies (Type II Anomalies). The final score combines contributions from original features and PCs.

    Parameters
    ----------
    plus : bool, optional
        If True, applies PCA and concatenates transformed features. Default is False.

    Attributes
    ----------
    pca_model : PCA or None
        PCA model for dimensionality reduction if `plus` is True.
    plus : bool
        Indicates whether PCA is applied.

    References
    ----------
    Aryal, Sunil & Ting, Kai & Haffari, Gholamreza. (2016). Revisiting Attribute Independence Assumption in Probabilistic Unsupervised Anomaly Detection.
    https://www.researchgate.net/publication/301610958_Revisiting_Attribute_Independence_Assumption_in_Probabilistic_Unsupervised_Anomaly_Detection

    Aryal, Sunil & Agrahari Baniya, Arbind & Santosh, Kc. (2019). Improved histogram-based anomaly detector with the extended principal component features.
    https://www.researchgate.net/publication/336132587_Improved_histogram-based_anomaly_detector_with_the_extended_principal_component_features
    """

    def __init__(self, plus=False):
        self.pca_model = None
        self.plus = plus
        super().__init__()

    def fit(
        self,
        X: np.ndarray,
        nbins: Union[int, str] = 5,
        random_state: int = 42,
        method="stddev",
    ) -> "SPAD":
        """
        Fit the SPAD model to the data.

        Parameters
        ----------
        X : np.ndarray
            The input data to fit. Must be a numpy array.
        nbins : Union[int, str], optional
            The number of bins to use for discretizing continuous features. Default is 5.
        random_state : int, optional
            The random seed for reproducibility. Default is 42.
        method : str, optional
            The method to compute the interval for continuous features. Default is "stddev", based on the original paper.

        Returns
        -------
        SPAD
            The fitted SPAD model.

        Notes
        -----
        - The data types and column names are extracted and stored.
        - If `self.plus` is True, PCA is applied to the data, and the transformed features are concatenated (SPAD+).
        - For categorical features, relative frequencies are computed using Laplace smoothing.
        - For continuous features, the data is discretized into bins, and probabilities are computed.
        - The decision scores are computed and stored in `self.decision_scores_`.
        """
        self._extract_feature_info(X)

        X = check_array(X)

        if self.plus:
            self.pca_model = PCA(random_state=random_state)
            self.pca_model = self.pca_model.fit(X)
            X = np.concatenate((X, self.pca_model.transform(X)), axis=1)
            self.feature_dtypes = np.concatenate(
                (self.feature_dtypes, np.array([np.float64] * len(self.feature_dtypes)))
            )

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
            else:
                lower_bound, upper_bound = StatisticalInterval.compute_interval(
                    X[:, i], method
                )
                bin_edges = np.linspace(lower_bound, upper_bound, nbins + 1)
                digitized = np.digitize(X[:, i], bin_edges, right=True)
                unique_bins, counts = np.unique(digitized, return_counts=True)
                probabilities = (counts + 1) / (np.sum(counts) + len(unique_bins))
                self.feature_distributions.append([probabilities, bin_edges])

        self.decision_scores_ = self._compute_decision_scores(X)
        return self

    def _compute_outlier_score(self, X: np.ndarray, i: int) -> np.ndarray:
        """
        Compute the outlier score for a specific feature column.
        """

        if isinstance(self.feature_dtypes[i], pd.CategoricalDtype):
            densities = np.array(
                [self.feature_distributions[i].get(value, 1e-9) for value in X[:, i]]
            )
        else:
            probabilities, bin_edges = self.feature_distributions[i]
            digitized = np.digitize(X[:, i], bin_edges, right=True)
            bin_indices = np.clip(digitized - 1, 0, len(probabilities) - 1)
            densities = probabilities[bin_indices]

        return np.log(densities + 1e-9)

    def _compute_decision_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Compute decision scores for the input data.
        """

        X = check_array(X)
        outlier_scores = np.zeros(shape=(X.shape[0], self.n_features))

        for i in range(self.n_features):
            outlier_scores[:, i] = self._compute_outlier_score(X, i)

        return np.sum(outlier_scores, axis=1).ravel()

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute decision function values for the input data.
        """

        self._check_columns(X)

        X = check_array(X)

        if self.plus:
            X = np.concatenate((X, self.pca_model.transform(X)), axis=1)
        return self._compute_decision_scores(X)
