import numpy as np
from scipy.stats import norm


def bootstrapping_bca(
    data,
    confidence_level=0.997,
    statistic=np.mean,
    n_resamples=1000,
    random_state=42,
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


def generate(
    df,
    confidence_level,
    statistic,
    n_resamples,
    random_state=42,
):
    """
    Calculate statistics for the reference distances, including confidence intervals and thresholds.
    """

    ci_lower, ci_upper = bootstrapping_bca(
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
