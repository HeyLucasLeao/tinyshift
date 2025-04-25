import numpy as np
from typing import Callable, Union, Tuple
from scipy.stats import norm


class BootstrapBCA:
    def __init__(self, random_state: int = 42):
        """
        Initialize the BootstrapBCA class.

        Parameters:
        - random_state (int): Random seed for reproducibility. Default is 42.
        """
        self.random_state = random_state

    def _jackknife_acceleration(self, data: np.ndarray, statistic: Callable) -> float:
        """Calculate the acceleration parameter using jackknife resampling."""
        n = len(data)
        jackknife = np.array([statistic(np.delete(data, i)) for i in range(n)])
        jackknife_mean = jackknife.mean()
        diffs = jackknife - jackknife_mean
        acceleration = np.sum(diffs**3) / (6.0 * (np.sum(diffs**2) ** 1.5))
        return acceleration

    def _bootstrap_statistics(
        self, data: np.ndarray, statistic: Callable, n_resamples: int
    ) -> np.ndarray:
        """Perform bootstrap resampling and calculate statistics."""
        rng = np.random.RandomState(self.random_state)
        return np.array(
            [
                statistic(rng.choice(data, size=len(data), replace=True))
                for _ in range(n_resamples)
            ]
        )

    @classmethod
    def compute_interval(
        cls,
        data: Union[np.ndarray, list],
        confidence_level: float,
        statistic: Callable,
        n_resamples: int,
        random_state: int = 42,
    ) -> Tuple[float, float]:
        """
        Calculates the bias-corrected and accelerated (BCa) bootstrap confidence interval for the given data.

        Parameters:
        - data (list or numpy array): Sample data.
        - confidence_level (float): Desired confidence level (e.g., 0.95 for 95%).
        - statistic (function): Statistical function to apply to the data. Default is np.mean.
        - n_resamples (int): Number of bootstrap resamples to perform. Default is 1000.

        Returns:
        - tuple: A tuple containing the lower and upper bounds of the BCa confidence interval.
        """
        instance = cls(random_state=random_state)
        data = np.asarray(data)

        # Bootstrap resampling
        sample_statistics = instance._bootstrap_statistics(data, statistic, n_resamples)

        # Jackknife resampling for acceleration
        acceleration = instance._jackknife_acceleration(data, statistic)

        # Bias correction
        observed_stat = statistic(data)
        bias = np.mean(sample_statistics < observed_stat)
        z0 = norm.ppf(bias)

        # Adjusting percentiles
        alpha = 1 - confidence_level
        z_alpha = norm.ppf(1 - alpha / 2)

        z_lower_bound = (z0 - z_alpha) / (1 - acceleration * (z0 - z_alpha)) + z0
        z_upper_bound = (z0 + z_alpha) / (1 - acceleration * (z0 + z_alpha)) + z0

        alpha_lower = norm.cdf(z_lower_bound)
        alpha_upper = norm.cdf(z_upper_bound)

        # Calculate lower and upper bounds from the percentiles
        lower_bound = np.quantile(sample_statistics, alpha_lower)
        upper_bound = np.quantile(sample_statistics, alpha_upper)

        return lower_bound, upper_bound
