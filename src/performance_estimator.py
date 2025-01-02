import numpy as np


class PerformanceEstimator:
    def __init__(self):
        pass

    def estimate_confusion_matrix(self, y_prob, y_pred):
        """
        y_prob: array-like, probabilidades previstas
        y_pred: array-like, classes previstas (0 ou 1)
        """
        # Probabilidade de predição estar errada
        p_wrong = np.abs(y_pred - y_prob)

        # Probabilidade de predição estar correta
        p_correct = 1 - p_wrong

        # Estimando TPs e TNs
        tp = np.sum(p_correct * (y_pred == 1))
        tn = np.sum(p_correct * (y_pred == 0))
        fn = np.sum(p_wrong * (y_pred == 0))
        fp = np.sum(p_wrong * (y_pred == 0))

        return tn, fp, fn, tp

    def sensitivity(self, y_prob, y_pred):
        _, _, fn, tp = self.estimate_confusion_matrix(y_prob, y_pred)
        return tp / (tp + fn)

    # Especificidade
    def specifity(self, y_prob, y_pred):
        tn, fp, _, _ = self.estimate_confusion_matrix(y_prob, y_pred)
        return tn / (tn + fp)

    # Precisão
    def precision_score(self, y_prob, y_pred):
        _, fp, _, tp = self.estimate_confusion_matrix(y_prob, y_pred)
        return tp / (tp + fp)

    # Função para calcular o F1 Score
    def f1_score(self, y_prob, y_pred):
        precision = self.precision_score(y_prob, y_pred)
        sensitivity = self.sensitivity(y_prob, y_pred)

        # Média harmônica entre Precisão e Sensibilidade
        return 2 * (precision * sensitivity) / (precision + sensitivity)

    def balanced_accuracy_score(self, y_prob, y_pred):
        sensitivity = self.sensitivity(y_prob, y_pred)
        specifity = self.specifity(y_prob, y_pred)
        return (sensitivity + specifity) / 2
