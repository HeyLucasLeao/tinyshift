from . import plot
import numpy as np
from scipy.stats import norm


class BaseModel:
    def __init__(
        self,
        reference_distribution,
        confidence_level,
        statistic,
        n_resamples,
        random_state,
        drift_limit,
    ):
        """
        Initializes the BaseModel class with reference distribution, statistics, and drift limits.

        Parameters:
        - reference_distribution (pd.DataFrame): Data containing the reference distribution.
        - confidence_level (float): Desired confidence level for statistical calculations.
        - statistic (function): Function to compute statistics.
        - n_resamples (int): Number of bootstrap resamples.
        - random_state (int): Seed for reproducibility.
        - drift_limit (str or tuple): Method or custom limits for drift thresholding.
        """
        self.statistics = self._statistic_generate(
            reference_distribution,
            confidence_level,
            statistic,
            n_resamples,
            random_state,
        )
        self.plot = plot.Plot(self.statistics, reference_distribution)
        self._drift_limit_generate(self.statistics, reference_distribution, drift_limit)

    def _bootstrapping_bca(
        self,
        data,
        confidence_level,
        statistic,
        n_resamples,
        random_state,
    ):
        """
        Calculates the bias-corrected and accelerated (BCa) bootstrap confidence interval for the given data.

        Parameters:
        - data (list or numpy array): Sample data.
        - confidence_level (float): Desired confidence level (e.g., 0.95 for 95%).
        - statistic (function): Statistical function to apply to the data. Default is np.mean.
        - n_resamples (int): Number of bootstrap resamples to perform. Default is 1000.
        - random_state (int): Random seed for reproducibility.

        Returns:
        - tuple: A tuple containing the lower and upper bounds of the BCa confidence interval.
        """
        np.random.seed(random_state)
        data = np.asarray(data)
        n = len(data)

        def generate_acceleration(data):
            """
            Calculates the acceleration parameter using jackknife resampling.
            """
            jackknife = np.zeros(n)
            for i in range(n):
                jackknife_sample = np.delete(data, i)
                jackknife[i] = statistic(jackknife_sample)

            jackknife_mean = np.mean(jackknife)
            jackknife_diffs = jackknife - jackknife_mean
            acceleration = np.sum(jackknife_diffs**3) / (
                6.0 * (np.sum(jackknife_diffs**2) ** 1.5)
            )
            return acceleration

        def calculate_bootstrap_statistics(data, statistic, n_resamples):
            """
            Performs bootstrap resampling and calculates the specified statistic.
            """
            return np.array(
                [
                    statistic(np.random.choice(data, size=n, replace=True))
                    for _ in range(n_resamples)
                ]
            )

        # Bootstrap resampling
        sample_statistics = calculate_bootstrap_statistics(data, statistic, n_resamples)

        # Jackknife resampling for acceleration
        acceleration = generate_acceleration(data)

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

    def _statistic_generate(
        self,
        df,
        confidence_level,
        statistic,
        n_resamples,
        random_state,
    ):
        """
        Calculate statistics for the reference distances, including confidence intervals and thresholds.
        """
        ci_lower, ci_upper = self._bootstrapping_bca(
            df["metric"],
            confidence_level,
            statistic,
            n_resamples,
            random_state,
        )
        estimated_mean = np.mean(df["metric"])

        return {
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "mean": estimated_mean,
        }

    def _calculate_limit(self, estimated_mean, std_deviation, factor=3):
        """Calculates the lower and upper limits based on the given factor."""
        lower_limit = estimated_mean - (factor * std_deviation)
        upper_limit = estimated_mean + (factor * std_deviation)
        return lower_limit, upper_limit

    def _deviation_threshold(self, df):
        """Calculates thresholds using standard deviation."""
        std_deviation = df["metric"].std()
        estimated_mean = df["metric"].mean()
        return self._calculate_limit(estimated_mean, std_deviation)

    def _mad_threshold(self, df):
        """Calculates thresholds using Median Absolute Deviation (MAD)."""
        mad_value = np.median(np.abs(df["metric"] - np.median(df["metric"])))
        estimated_mean = df["metric"].mean()
        return self._calculate_limit(estimated_mean, mad_value)

    def _drift_limit_generate(self, statistics, distribution, drift_limit):
        """
        Determines the lower and upper drift limits based on different threshold methods.
        """
        if isinstance(drift_limit, str):
            if drift_limit == "deviation":
                lower_limit, upper_limit = self._deviation_threshold(distribution)
            elif drift_limit == "mad":
                lower_limit, upper_limit = self._mad_threshold(distribution)
            else:
                raise ValueError(f"Unsupported drift limit method: {drift_limit}")
        elif isinstance(drift_limit, tuple) and len(drift_limit) == 2:
            lower_limit, upper_limit = drift_limit
        else:
            raise ValueError(
                "Drift limit must be a string or a tuple with two elements."
            )

        # Update the statistics dictionary with the new thresholds
        statistics["lower_limit"] = lower_limit
        statistics["upper_limit"] = upper_limit
