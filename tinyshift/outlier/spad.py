import numpy as np
import pandas as pd
from sklearn.utils import check_array
from collections import Counter
from sklearn.decomposition import PCA
from .base import BaseHistogramModel
from typing import Union


class SPAD(BaseHistogramModel):

    def __init__(self, plus=False):
        self.pca_model = None
        self.plus = plus
        super().__init__()

    def _check_bins(self, X: np.ndarray, nbins: Union[int, str]) -> int:
        """
        Valida e determina o número de bins para construção do histograma.

        Parâmetros:
        -----------
        X : array-like, shape (n_samples,)
            Dados de entrada para uma única característica.

        nbins : int ou str
            Número de bins ou uma estratégia válida de binning.

        Retorno:
        --------
        int
            Número de bins a ser usado.
        """
        if isinstance(nbins, int) and nbins > 0:
            return nbins
        elif isinstance(nbins, str):
            bin_edges = np.histogram_bin_edges(X, bins=nbins)
            return len(bin_edges) - 1
        else:
            raise ValueError(
                "nbins deve ser um número inteiro positivo ou uma estratégia válida de binning."
            )

    def fit(
        self,
        X: np.ndarray,
        nbins: Union[int, str] = 5,
        random_state: int = 42,
    ) -> "SPAD":
        """
        Treina o modelo Histogram aprendendo as distribuições das características.

        Parâmetros:
        -----------
        X : array-like, shape (n_samples, n_features)
            Dados de entrada contendo características categóricas e/ou contínuas.

        nbins : int ou str, padrão=10
            Número de bins para discretização ou uma estratégia válida de binning.

        Retorno:
        --------
        self : object
            A instância do modelo ajustado.
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
        Calcula a densidade (probabilidade) de cada amostra de acordo com a distribuição da característica.

        Parâmetros:
        -----------
        X : array-like, shape (n_samples, n_features)
            Dados de entrada.

        i : int
            Índice da característica a ser processada.

        Retorno:
        --------
        density : np.ndarray
            Densidade calculada para a característica i.
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
        Calcula o score de anomalia para cada instância no conjunto de dados.

        Parâmetros:
        -----------
        X : array-like, shape (n_samples, n_features)
            Conjunto de dados a ser avaliado quanto a anomalias.

        Retorno:
        --------
        scores : array, shape (n_samples,)
            Scores de anomalia para o conjunto de dados. Scores maiores indicam maior probabilidade de serem outliers.
        """
        X = check_array(X)
        outlier_scores = np.zeros(shape=(X.shape[0], self.n_features))

        for i in range(self.n_features):
            outlier_scores[:, i] = self._compute_outlier_score(X, i)

        return np.sum(outlier_scores, axis=1).ravel()

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula o score de anomalia para um conjunto de dados, considerando a transformação PCA se `plus=True`.

        Parâmetros:
        -----------
        X : array-like, shape (n_samples, n_features)
            Conjunto de dados a ser avaliado quanto a anomalias.

        Retorno:
        --------
        scores : array, shape (n_samples,)
            Scores de anomalia para o conjunto de dados.
        """
        self._check_columns(X)

        X = check_array(X)

        if self.plus:
            X = np.concatenate((X, self.pca_model.transform(X)), axis=1)
        return self._decision_function(X)
