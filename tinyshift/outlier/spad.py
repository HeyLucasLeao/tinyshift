import numpy as np
import pandas as pd
from sklearn.utils import check_array
from collections import Counter
from sklearn.decomposition import PCA
from .base import BaseHistogramModel
from typing import Union


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
    dtypes : np.ndarray
        Data types of input features.
    columns : Index
        Column names of the input data (if a pandas DataFrame).
    n_features : int
        Number of input features.
    dist_ : list
        Feature distributions (relative frequencies for categorical or probabilities and bin edges for continuous features).
    decision_scores_ : np.ndarray
        Computed anomaly scores for input data.

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
    ) -> "SPAD":
        """
        Fit the SPAD model to the data.
        Parameters
        ----------
        X : np.ndarray
            The input data to fit. Can be a numpy array, pandas Series, or pandas DataFrame.
        nbins : Union[int, str], optional
            The number of bins to use for discretizing continuous features. Default is 5.
        random_state : int, optional
            The random seed for reproducibility. Default is 42.
        Returns
        -------
        SPAD
            The fitted SPAD model.
        Notes
        -----
        - If `X` is a pandas Series or DataFrame, the data types and column names are stored.
        - If `self.plus` is True, PCA is applied to the data and the transformed features are concatenated. (SPAD+)
        - For categorical features, relative frequencies are computed using Laplace smoothing.
        - For continuous features, the data is discretized into bins and probabilities are computed.
        - The decision scores are computed and stored in `self.decision_scores_`.
        """
        if isinstance(X, (pd.Series, pd.DataFrame)):
            self.dtypes = np.asarray(X.dtypes)
            self.columns = X.columns

        X = check_array(X)

        if self.plus:
            self.pca_model = PCA(random_state=random_state)
            self.pca_model = self.pca_model.fit(X)
            X = np.concatenate((X, self.pca_model.transform(X)), axis=1)
            self.dtypes = np.concatenate(
                (self.dtypes, np.array([np.float64] * len(self.dtypes)))
            )

        _, self.n_features = X.shape

        for i in range(self.n_features):
            nbins = self._check_bins(X[:, i], nbins)

            if isinstance(self.dtypes[i], pd.CategoricalDtype):
                counts = Counter(X[:, i])
                total_count = sum(counts.values())
                relative_freq = {
                    k: (v + 1) / (total_count + len(counts)) for k, v in counts.items()
                }
                self.dist_.append(relative_freq)
            else:
                mean = np.mean(X[:, i])
                std = np.std(X[:, i])
                lower_bound = mean - 3 * std
                upper_bound = mean + 3 * std
                bin_edges = np.linspace(lower_bound, upper_bound, nbins + 1)
                digitized = np.digitize(X[:, i], bin_edges, right=True)
                unique_bins, counts = np.unique(digitized, return_counts=True)
                probabilities = (counts + 1) / (np.sum(counts) + len(unique_bins))
                self.dist_.append([probabilities, bin_edges])

        self.decision_scores_ = self._decision_function(X)
        return self

    def _compute_outlier_score(self, X: np.ndarray, i: int) -> np.ndarray:
        """
        Compute the outlier score for a given feature column in the dataset.
        Parameters
        ----------
        X : np.ndarray
            The input data array.
        i : int
            The index of the feature column for which to compute the outlier score.
        Returns
        -------
        np.ndarray
            An array of outlier scores for the specified feature column.
        """

        if isinstance(self.dtypes[i], pd.CategoricalDtype):
            density = np.array([self.dist_[i].get(value, 1e-9) for value in X[:, i]])
        else:
            probabilities, bin_edges = self.dist_[i]
            digitized = np.digitize(X[:, i], bin_edges, right=True)
            bin_indices = np.clip(digitized - 1, 0, len(probabilities) - 1)
            density = probabilities[bin_indices]

        return np.log(density + 1e-9)

    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the decision function for the input data X.
        This function calculates the outlier scores for each feature in the input
        data and returns the sum of these scores for each sample.
        Parameters
        ----------
        X : np.ndarray
            The input data array of shape (n_samples, n_features).
        Returns
        -------
        np.ndarray
            The computed outlier scores for each sample, as a 1D array of shape (n_samples,).
        """

        X = check_array(X)
        outlier_scores = np.zeros(shape=(X.shape[0], self.n_features))

        for i in range(self.n_features):
            outlier_scores[:, i] = self._compute_outlier_score(X, i)

        return np.sum(outlier_scores, axis=1).ravel()

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the decision function for the input data.
        Parameters
        ----------
        X : np.ndarray
            Input data array.
        Returns
        -------
        np.ndarray
            Decision function values for the input data.
        """

        self._check_columns(X)

        X = check_array(X)

        if self.plus:
            X = np.concatenate((X, self.pca_model.transform(X)), axis=1)
        return self._decision_function(X)
