from typing import List, Any, Optional
import pandas as pd
import numpy as np
import pickle
from itertools import product
from sklearn.base import BaseEstimator, TransformerMixin
from .encoder import TransactionEncoder


class TransactionAnalyzer(BaseEstimator, TransformerMixin):
    """
    TransactionAnalyzer for association rule mining and transaction pattern analysis.

    This class provides tools for analyzing transactional data, particularly for
    market basket analysis and association rule mining. It encodes transaction data
    into a one-hot format and calculates various association metrics for measuring
    the strength and direction of relationships between items.

    The analyzer is designed to work with transactional data where each transaction
    is a list of items. It supports multiple association metrics including Zhang's metric,
    conviction, confidence and certainty factor.

    Attributes:
        encoder_ (TransactionEncoder): Encoder for transaction data
        columns_ (List[str]): Column names after encoding
        transactions_ (pd.DataFrame): Encoded transactions dataframe
    """

    def __init__(self) -> None:
        """Initialize TransactionAnalyzer.

        Attributes:
            encoder (TransactionEncoder): Encoder for transaction data
            columns_ (List[str]): Column names after encoding
            transactions (pd.DataFrame): Encoded transactions dataframe
        """
        self.encoder_: Optional[TransactionEncoder] = None
        self.columns_: Optional[List[str]] = None
        self.transactions_: Optional[pd.DataFrame] = None

    def fit(self, transactions: List[List[Any]]) -> "TransactionAnalyzer":
        """
        Fit the encoder to transactions and create encoded dataframe.

        Parameters
        ----------
            transactions : List of transactions where each transaction is a list of items

        Returns
        -------
            self: Fitted TransactionAnalyzer instance

        Raises
        -------
            ValueError: If transactions is empty or invalid
        """
        self.encoder_ = TransactionEncoder()
        self.encoder_.fit(transactions)
        self.columns_ = self.encoder_.columns_
        self.transactions_ = pd.DataFrame(
            self.encoder_.transform(transactions), columns=self.columns_
        )
        return self

    def transform(self, transactions: List[List[Any]]) -> np.ndarray:
        """Transform transactions to one-hot encoding."""
        if self.encoder_ is None:
            raise ValueError("Analyzer must be fitted before transforming data")
        return self.encoder_.transform(transactions)

    def fit_transform(self, transactions: List[List[Any]]) -> np.ndarray:
        """Fit and transform transactions."""
        self.fit(transactions)
        return self.transform(transactions)

    def save(self, filename: str) -> None:
        """Save analyzer to file using pickle."""
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename: str) -> "TransactionAnalyzer":
        """Load analyzer from file."""
        with open(filename, "rb") as f:
            return pickle.load(f)

    def _get_support(self, items: pd.Series) -> float:
        """Calculate support for an itemset."""
        return items.mean()

    def _get_counts(self, items: pd.Series) -> int:
        """
        Get counts needed for various association measures.
        """
        return items.sum()

    def confidence(self, antecedent: str, consequent: str) -> float:
        """
        Calculate confidence metric for association rules.

        Confidence measures the probability that consequent occurs given that antecedent occurs.
        Values range from 0 to 1, where higher values indicate stronger rules.

        Parameters
        ----------
            antecedent : The antecedent item in the association rule
            consequent : The consequent item in the association rule

        Returns
        ----------
            float: Confidence value between 0 and 1

        Raises
        ----------
            ValueError: If analyzer has not been fitted
            KeyError: If either antecedent or consequent item is not found in encoded transactions
        """
        if self.transactions_ is None:
            raise ValueError("Analyzer must be fitted before calculating confidence")

        try:
            antecedent_series = self.transactions_[antecedent]
            consequent_series = self.transactions_[consequent]
        except KeyError as e:
            raise KeyError(f"Item not found in encoded transactions: {e}")

        supportA = self._get_support(antecedent_series)
        supportAC = self._get_support(
            np.logical_and(antecedent_series, consequent_series)
        )

        if supportA == 0:
            return 0.0

        return supportAC / supportA

    def conviction(self, antecedent: str, consequent: str) -> float:
        """
        Calculate conviction metric for association rules.

        Conviction measures the degree of implication of a rule.
        It represents how often the rule would be incorrect if the items were independent.

        - conviction = 1 : items are independent
        - conviction > 1 : positive correlation (higher is better)
        - conviction → ∞ : perfect implication

        Parameters
        ----------
            antecedent : The antecedent item in the association rule
            consequent : The consequent item in the association rule

        Returns
        ----------
            float : Conviction metric value between 1 and ∞

        Raises:
            ValueError: If analyzer has not been fitted
            KeyError: If either antecedent or consequent item is not found in encoded transactions
        """
        if self.transactions_ is None:
            raise ValueError("Analyzer must be fitted before calculating conviction")

        try:
            antecedent_series = self.transactions_[antecedent]
            consequent_series = self.transactions_[consequent]
        except KeyError as e:
            raise KeyError(f"Item not found in encoded transactions: {e}")

        supportA = self._get_support(antecedent_series)
        supportC = self._get_support(consequent_series)
        supportAC = self._get_support(
            np.logical_and(antecedent_series, consequent_series)
        )

        if supportA == 0:
            return np.nan

        if supportC == 1 or supportAC == supportA * supportC:
            return 1.0

        confidence = supportAC / supportA
        denominator = 1 - confidence

        if denominator == 0:
            return np.inf

        return (1 - supportC) / denominator

    def zhang_metric(self, antecedent: str, consequent: str) -> float:
        """
        Calculate Zhang's metric for association rule mining.

        Zhang's metric measures the strength of association between two items.
        Values range from -1 to 1, where positive values indicate positive association.

        Parameters
        ----------
            antecedent : The antecedent item in the association rule
            consequent : The consequent item in the association rule

        Returns
        ----------
            float : Zhang's metric value between -1 and 1

        Raises:
            ValueError: If analyzer has not been fitted
            KeyError: If either antecedent or consequent item is not found in encoded transactions
        """
        if self.transactions_ is None:
            raise ValueError(
                "Analyzer must be fitted before calculating Zhang's metric"
            )

        try:
            antecedent_series = self.transactions_[antecedent]
            consequent_series = self.transactions_[consequent]
        except KeyError as e:
            raise KeyError(f"Item not found in encoded transactions: {e}")

        supportA = self._get_support(antecedent_series)
        supportC = self._get_support(consequent_series)
        supportAC = self._get_support(
            np.logical_and(antecedent_series, consequent_series)
        )

        numerator = supportAC - supportA * supportC
        denominator = max(supportAC * (1 - supportA), supportA * (supportC - supportAC))

        return numerator / denominator if denominator != 0 else 0.0

    def leverage(self, antecedent: str, consequent: str) -> float:
        """
        Calculate leverage metric for association rules.

        Leverage measures the difference between the observed frequency of co-occurrence and the frequency expected
        if the items were independent. Values range from -0.25 and 0.25, where positive values indicate positive association.

        Parameters
        ----------
            antecedent : The antecedent item in the association rule
            consequent : The consequent item in the association rule

        Returns
        ----------
            float : Leverage value between -0.25 and 0.25

        Raises:
            ValueError: If analyzer has not been fitted
            KeyError: If either antecedent or consequent item is not found in encoded transactions
        """
        if self.transactions_ is None:
            raise ValueError("Analyzer must be fitted before calculating leverage")

        try:
            antecedent_series = self.transactions_[antecedent]
            consequent_series = self.transactions_[consequent]
        except KeyError as e:
            raise KeyError(f"Item not found in encoded transactions: {e}")

        supportA = self._get_support(antecedent_series)
        supportC = self._get_support(consequent_series)
        supportAC = self._get_support(
            np.logical_and(antecedent_series, consequent_series)
        )

        return supportAC - (supportA * supportC)

    def cosine(self, antecedent: str, consequent: str) -> float:
        """
        Calculate cosine similarity for association rules.

        Cosine similarity measures the cosine of the angle between the two item vectors.
        It is equivalent to the support of the co-occurrence divided by the
        geometric mean of the individual supports.

        Values range from 0 to 1, where higher values indicate stronger association.

        Parameters
        ----------
            antecedent : The antecedent item in the association rule
            consequent : The consequent item in the association rule

        Returns
        ----------
            float : Cosine similarity value between 0 and 1

        Raises:
            ValueError: If analyzer has not been fitted
            KeyError: If either antecedent or consequent item is not found in encoded transactions
        """
        if self.transactions_ is None:
            raise ValueError("Analyzer must be fitted before calculating cosine")

        try:
            antecedent_series = self.transactions_[antecedent]
            consequent_series = self.transactions_[consequent]
        except KeyError as e:
            raise KeyError(f"Item not found in encoded transactions: {e}")

        supportA = self._get_support(antecedent_series)
        supportC = self._get_support(consequent_series)
        supportAC = self._get_support(
            np.logical_and(antecedent_series, consequent_series)
        )

        return supportAC / np.sqrt(supportA * supportC)

    def yules_q(self, antecedent: str, consequent: str) -> float:
        """
        Calculate Yule's Q coefficient for association rules.

        Yule's Q is a measure of association between two binary variables based on
        the odds ratio. It ranges from -1 (perfect negative association) to +1
        (perfect positive association), with 0 indicating no association.

        Parameters
        ----------
            antecedent : The antecedent item in the association rule
            consequent : The consequent item in the association rule

        Returns
        ----------
            float : Yule's Q coefficient between -1 and 1

        Raises:
            ValueError: If analyzer has not been fitted
            KeyError: If either antecedent or consequent item is not found
        """

        if self.transactions_ is None:
            raise ValueError("Analyzer must be fitted before calculating cosine")

        try:
            antecedent_series = self.transactions_[antecedent]
            consequent_series = self.transactions_[consequent]
        except KeyError as e:
            raise KeyError(f"Item not found in encoded transactions: {e}")

        nX = self._get_counts(antecedent_series)
        nY = self._get_counts(consequent_series)
        nXY = self._get_counts(np.logical_and(antecedent_series, consequent_series))

        odds_ratio = (nXY * (len(self.transactions_) - nX - nY + nXY)) / (
            (nX - nXY) * (nY - nXY)
        )

        return (odds_ratio - 1) / (odds_ratio + 1)

    def correlation_matrix(
        self,
        row_items: List[str],
        column_items: List[str],
        metric: str = "zhang",
    ) -> pd.DataFrame:
        """
        Create a correlation matrix between row and column items using Zhang's metric.

        Parameters
        -----------
        row_items : List[str]
            List of items to use as rows in the correlation matrix
        column_items : List[str]
            List of items to use as columns in the correlation matrix

        Returns
        --------
        pd.DataFrame
            Correlation matrix with row_items as index, column_items as columns,
            and Zhang's metric values as cells

        Raises
        -------
        ValueError
            If analyzer has not been fitted
        """
        if self.transactions_ is None:
            raise ValueError(
                "Analyzer must be fitted before creating correlation matrix"
            )

        metric_mapping = {
            "zhang": self.zhang_metric,
            "conviction": self.conviction,
            "confidence": self.confidence,
            "leverage": self.leverage,
            "cosine": self.cosine,
            "yules_q": self.yules_q,
        }

        callable_function = metric_mapping.get(metric, None)

        if not callable_function:
            raise ValueError(
                f"Unknown metric: '{metric}'. Available metrics: {metric_mapping.keys()}"
            )

        pairs = list(product(row_items, column_items))

        metric_values = [callable_function(row, columns) for row, columns in pairs]

        index = pd.MultiIndex.from_tuples(pairs)

        metric = pd.DataFrame(metric_values, index=index).unstack()

        metric.columns = metric.columns.droplevel()

        return metric
