from typing import Union, List, Dict
import numpy as np
import pandas as pd


class BaseHistogramModel:
    """
    Base class for histogram-based models.

    Attributes:
    n_features : int or None
        Number of features in the dataset.

    dtypes : list or None
        Data types of the features.

    columns : list
        List of column names.

    dist_ : list
        List of distributions for each feature. Each distribution can be a dictionary or a list of numpy arrays.

    decision_scores_ : array-like or None
        Decision scores for the samples.
    """

    def __init__(self):
        self.n_features = None
        self.dtypes = None
        self.columns = []
        self.dist_: List[Union[Dict, List[np.ndarray]]] = []
        self.decision_scores_ = None

    def _check_bins(self, X: np.ndarray, nbins: Union[int, str]) -> int:
        """
        Determine the number of bins for histogram binning.
        Parameters:
        -----------
        X : np.ndarray
            The input data array for which the bins are to be determined.
        nbins : Union[int, str]
            The number of bins or a binning strategy. If an integer, it must be positive.
            If a string, it should be a valid binning strategy recognized by `np.histogram_bin_edges`.
        Returns:
        --------
        int
            The number of bins to be used for histogram binning.
        Raises:
        -------
        ValueError
            If `nbins` is not a positive integer or a valid binning strategy.
        """

        if isinstance(nbins, int) and nbins > 0:
            return nbins
        elif isinstance(nbins, str):
            bin_edges = np.histogram_bin_edges(X, bins=nbins)
            return len(bin_edges) - 1
        else:
            raise ValueError(
                "nbins must be a positive integer or a valid `np.histogram_bin_edges` binning strategy."
            )

    def _check_columns(self, X: Union[np.ndarray, "pd.DataFrame"]):
        """
        Check if the columns of the input data match the columns of the training data.

        Parameters:
        -----------
        X : Union[np.ndarray, pd.DataFrame]
            The input data array or DataFrame to be checked.

        Raises:
        -------
        ValueError
            If the columns of the input data do not match the columns of the training data.
        """
        if isinstance(X, pd.DataFrame):
            if not all(X.columns == self.columns):
                raise ValueError(
                    "The columns of the input data do not match the columns of the training data."
                )
