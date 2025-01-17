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
    Methods
    -------
    __init__():
        Initializes the HBOS model.
    _compute_density(X: np.ndarray, i: int, dynamic_bins: bool, nbins: int) -> np.ndarray:
        Calculates the density for a feature, considering dynamic or static bins.
    fit(X: np.ndarray, nbins: Union[int, str] = 10, dynamic_bins: bool = False) -> "HBOS":
        Trains the Histogram model by learning the distributions of the features.
    _compute_outlier_score(X: np.ndarray, i: int) -> np.ndarray:
        Calculates the outlier score for a specific feature.
    decision_function(X: np.ndarray) -> np.ndarray:
        Calculates the outlier score for each instance in the dataset.
    References
    ----------
    Goldstein, Markus & Dengel, Andreas. (2012). Histogram-based Outlier Score (HBOS): A fast Unsupervised Anomaly Detection Algorithm.
    https://www.researchgate.net/publication/231614824_Histogram-based_Outlier_Score_HBOS_A_fast_Unsupervised_Anomaly_Detection_Algorithm
    """

    def __init__(self):
        super().__init__()

    def _compute_density(
        self,
        X: np.ndarray,
        i: int,
        dynamic_bins: bool,
        nbins: int,
    ) -> np.ndarray:
        """
        Compute the density estimation for a given feature in the dataset.
        Parameters:
        -----------
        X : np.ndarray
            The input data array.
        i : int
            The index of the feature to compute the density for.
        dynamic_bins : bool
            If True, use dynamic binning based on percentiles.
        nbins : int
            The number of bins to use for the histogram.
        Returns:
        --------
        np.ndarray
            If the feature is categorical, returns a dictionary of relative frequencies.
            If dynamic_bins is True, returns a list containing densities and bin edges computed using dynamic binning.
            Otherwise, returns a list containing densities and bin edges computed using fixed binning.
        """
        if isinstance(self.dtypes[i], pd.CategoricalDtype):
            counts = Counter(X[:, i])
            total_count = sum(counts.values())
            relative_freq = {
                k: (v + 1) / (total_count + len(counts)) for k, v in counts.items()
            }
            return relative_freq
        elif dynamic_bins:
            percentiles = np.percentile(X[:, i], q=np.linspace(0, 100, nbins + 1))
            bin_edges = np.unique(percentiles)
            densities, _ = np.histogram(X[:, i], bins=bin_edges, density=True)
            return [densities, bin_edges]
        else:
            densities, bin_edges = np.histogram(X[:, i], bins=nbins, density=True)
            return [densities, bin_edges]

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
        """
        if isinstance(X, (pd.Series, pd.DataFrame)):
            self.dtypes = np.asarray(X.dtypes)
            self.columns = X.columns

        X = check_array(X)
        _, self.n_features = X.shape

        for i in range(self.n_features):
            nbins = self._check_bins(X[:, i], nbins)
            self.dist_.append(self._compute_density(X, i, dynamic_bins, nbins))

        self.decision_scores_ = self.decision_function(X)
        return self

    def _compute_outlier_score(self, X: np.ndarray, i: int) -> np.ndarray:
        """
        Compute the outlier score for a given feature in the dataset.
        Parameters:
        -----------
        X : np.ndarray
            The input data array.
        i : int
            The index of the feature for which to compute the outlier score.
        Returns:
        --------
        np.ndarray
            The outlier scores for the specified feature.
        """

        if isinstance(self.dtypes[i], pd.CategoricalDtype):
            density = np.array([self.dist_[i].get(value, 1e-9) for value in X[:, i]])
        else:
            densities, bin_edges = self.dist_[i]
            bin_indices = np.searchsorted(bin_edges, X[:, i], side="right") - 1
            bin_indices = np.clip(bin_indices, 0, len(densities) - 1)
            density = densities[bin_indices]
        return -np.log(density + 1e-9)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the decision function for the input data X.
        Parameters
        ----------
        X : np.ndarray
            Input data to compute the decision function. Can be a numpy array or a pandas DataFrame.
        Returns
        -------
        np.ndarray
            An array of anomaly scores for each sample in the input data. The scores are negative, with lower values indicating higher likelihood of being an outlier.
        Raises
        ------
        ValueError
            If the input data is a pandas DataFrame and its columns do not match the columns of the training data.
        """

        self._check_columns(X)

        X = check_array(X)
        outlier_scores = np.zeros(shape=(X.shape[0], self.n_features))

        # Calcula o score de anomalia para cada característica
        for i in range(self.n_features):
            outlier_scores[:, i] = self._compute_outlier_score(X, i)

        return np.sum(outlier_scores, axis=1).ravel() * -1
