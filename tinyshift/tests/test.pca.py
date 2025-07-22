import unittest
import numpy as np
from tinyshift.outlier.pca import PCAReconstructionError


class TestPCAReconstructionError(unittest.TestCase):
    def setUp(self):
        self.model = PCAReconstructionError()

    def test_fit_sets_attributes(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        self.model.fit(X)
        self.assertTrue(hasattr(self.model, "PCA"))
        self.assertTrue(hasattr(self.model, "decision_scores_"))

    def test_score_output(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        self.model.fit(X)
        scores = self.model.decision_function(X)
        self.assertEqual(scores, self.model.decision_function)

    def test_decision_function_output(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        self.model.fit(X)
        self.assertTrue(self.model.predict(X).shape[0] == X.shape[0])

    def test_fit_not_fitted(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        with self.assertRaises(ValueError):
            self.model.predict(X)


if __name__ == "__main__":
    unittest.main()
