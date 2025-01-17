import unittest
import numpy as np
import pandas as pd
from tinyshift.outlier.hbos import HBOS


class TestHBOS(unittest.TestCase):

    def setUp(self):
        self.hbos = HBOS()
        self.hbos.feature_dtypes = np.array([pd.CategoricalDtype(), np.float64])
        self.hbos.feature_distributions = [
            {1: 0.2, 2: 0.3, 3: 0.5},
            [
                np.array([6.25, 0.0, 6.25, 0.0, 12.5]),
                np.array([0.1, 0.14, 0.18, 0.22, 0.26, 0.3]),
            ],
        ]
        self.hbos.n_features = 2

    def test_compute_outlier_score_categorical(self):
        X = np.array([[1, 0], [2, 0], [3, 0], [4, 0]])
        scores = self.hbos._compute_outlier_score(X, 0)
        expected_scores = -np.log(np.array([0.2, 0.3, 0.5, 1e-9]) + 1e-9)
        np.testing.assert_array_almost_equal(scores, expected_scores, decimal=3)

    def test_compute_outlier_score_continuous(self):
        X = np.array([[0, 0.1], [0, 0.2], [0, 0.3], [0, 0.3]])
        scores = self.hbos._compute_outlier_score(X, 1)
        expected_scores = -np.log(np.array([6.25, 6.25, 12.5, 12.5]) + 1e-9)
        np.testing.assert_array_almost_equal(scores, expected_scores, decimal=3)

    def test_fit(self):
        hbos = HBOS()
        X = np.array([[1, 0.1], [2, 0.2], [3, 0.3], [4, 0.4]])
        hbos.fit(X)
        self.assertIsNotNone(hbos.feature_dtypes)
        self.assertIsNotNone(hbos.feature_distributions)
        self.assertEqual(len(hbos.feature_dtypes), X.shape[1])
        self.assertEqual(len(hbos.feature_distributions), X.shape[1])
        self.assertTrue(hasattr(hbos, "decision_scores_"))
        self.assertEqual(hbos.n_features, X.shape[1])

    def test_dynamic_bins(self):
        hbos = HBOS()
        X = np.array([[1, 0.1], [2, 0.2], [3, 0.3], [4, 0.4], [5, 0.5]])
        hbos.fit(X, dynamic_bins=True)
        self.assertIsNotNone(hbos.feature_distributions)
        self.assertEqual(len(self.hbos.feature_distributions), X.shape[1])
        for distribution in hbos.feature_distributions:
            if isinstance(distribution, list):
                self.assertTrue(
                    all(isinstance(bin_edges, np.ndarray) for bin_edges in distribution)
                )

    def test_decision_function(self):
        X = np.array([[1, 0.1], [2, 0.2], [3, 0.3], [4, 0.3]])
        scores = self.hbos.decision_function(X)
        expected_scores = (
            -np.log(np.array([0.2, 0.3, 0.5, 1e-9]) + 1e-9)
            + -np.log(np.array([6.25, 6.25, 12.5, 12.5]) + 1e-9)
        ) * -1
        np.testing.assert_array_almost_equal(scores, expected_scores, decimal=3)


if __name__ == "__main__":
    unittest.main()
