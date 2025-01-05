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


def calculate_limits(estimated_mean, std_deviation, factor=3):
    lower_limit = estimated_mean - (factor * std_deviation)
    upper_limit = estimated_mean + (factor * std_deviation)
    return lower_limit, upper_limit


def deviation_threshold(df):
    std_deviation = df["metric"].std()
    estimated_mean = df["metric"].mean()
    return calculate_limits(estimated_mean, std_deviation)


def mad_threshold(df):
    mad_value = np.median(np.abs(df["metric"] - np.median(df["metric"])))
    estimated_mean = df["metric"].mean()
    return calculate_limits(estimated_mean, mad_value)


def calculate_statistics(
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


def generate_drift_limit(statistics, distribution, drift_limit):
    """
    Function to determine the lower and upper drift limit based on different threshold methods.

    Parameters:
    statistics (dict): Dictionary to store the calculated drift limit.
    distribution (array-like): Data distribution for analysis.
    drift limit (str or tuple): Threshold method to use or directly provided limits.

    Supported drift limits:
    - "deviation": Uses the standard deviation method.
    - "mad": Uses the Median Absolute Deviation (MAD) method.
    - tuple: Uses a custom limit.

    Raises:
    ValueError: If the provided threshold method is not supported.

    Returns:
    None: Updates the statistics dictionary with the calculated thresholds.
    """

    # Check if the threshold is a string or a tuple with limit values
    if isinstance(drift_limit, str):
        if drift_limit == "deviation":
            lower_limit, upper_limit = deviation_threshold(distribution)
        elif drift_limit == "mad":
            lower_limit, upper_limit = mad_threshold(distribution)
        else:
            raise ValueError(f"Unsupported drift limit method: {drift_limit}")
    elif isinstance(drift_limit, tuple) and len(drift_limit) == 2:
        lower_limit, upper_limit = drift_limit
    else:
        raise ValueError("Drift limit must be a string or a tuple with two elements.")

    # Update the statistics dictionary with the new thresholds
    statistics["lower_limit"] = lower_limit
    statistics["upper_limit"] = upper_limit
