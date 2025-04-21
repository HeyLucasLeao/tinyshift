from typing import Union, List, Dict
import numpy as np
import pandas as pd


class BaseHistogramModel:
    """
    Base class for histogram-based models.

    Attributes:
    n_features : int or None
        Number of features in the dataset.

    feature_dtypes : list or None
        Data types of the features.

    feature_names : list
        List of column names.

    feature_distributions : list
        List of distributions for each feature. Each distribution can be a dictionary or a list of numpy arrays.

    decision_scores_ : array-like or None
        Decision scores for the samples.
    """

    def __init__(self):
        self.n_features = None
        self.feature_names = None
        self.feature_dtypes = []
        self.feature_distributions: List[Union[Dict, List[np.ndarray]]] = []
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
            try:
                bin_edges = np.histogram_bin_edges(X, bins=nbins)
                return len(bin_edges) - 1
            except ValueError as e:
                raise ValueError(
                    f"Invalid binning strategy '{nbins}'. Please use a positive integer or one of the following valid strategies: "
                    "'auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', or 'sqrt'.\n"
                    "Descriptions:\n"
                    "- 'auto': Minimum bin width between the 'sturges' and 'fd' estimators. Provides good all-around performance.\n"
                    "- 'fd' (Freedman Diaconis Estimator): Robust estimator that accounts for data variability and size.\n"
                    "- 'doane': Improved version of Sturges’ estimator for non-normal datasets.\n"
                    "- 'scott': Less robust estimator that considers data variability and size.\n"
                    "- 'stone': Based on leave-one-out cross-validation of the integrated squared error. Generalizes Scott’s rule.\n"
                    "- 'rice': Considers only data size, often overestimates the number of bins.\n"
                    "- 'sturges': Optimal for Gaussian data, underestimates bins for large non-Gaussian datasets.\n"
                    "- 'sqrt': Square root of data size, used for simplicity and speed."
                ) from e
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
            if not all(X.columns == self.feature_names):
                raise ValueError(
                    "The columns of the input data do not match the columns of the training data."
                )

    def _extract_feature_info(self, X: Union[pd.Series, pd.DataFrame]):
        """
        Extract feature information from the input data.

        Parameters:
        -----------
        X : Union[pd.Series, pd.DataFrame]
            The input data from which to extract feature information.

        Raises:
        -------
        TypeError
            If the input data is not a pandas Series or DataFrame.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            self.feature_dtypes = X.dtypes.values
        elif isinstance(X, pd.Series):
            self.feature_names = [X.name] if X.name else ["feature_0"]
            self.feature_dtypes = [X.dtype]
        elif isinstance(X, np.ndarray):
            if X.ndim == 1:
                self.feature_names = ["feature_0"]
                self.feature_dtypes = [X.dtype]
            else:
                self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
                self.feature_dtypes = np.repeat(X.dtype, X.shape[1])
        else:
            raise TypeError(
                "Input data must be a pandas Series, DataFrame, or numpy ndarray."
            )
