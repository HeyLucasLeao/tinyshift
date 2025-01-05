import numpy as np


def calculate_limit(estimated_mean, std_deviation, factor=3):
    lower_limit = estimated_mean - (factor * std_deviation)
    upper_limit = estimated_mean + (factor * std_deviation)
    return lower_limit, upper_limit


def deviation_threshold(df):
    std_deviation = df["metric"].std()
    estimated_mean = df["metric"].mean()
    return calculate_limit(estimated_mean, std_deviation)


def mad_threshold(df):
    mad_value = np.median(np.abs(df["metric"] - np.median(df["metric"])))
    estimated_mean = df["metric"].mean()
    return calculate_limit(estimated_mean, mad_value)


def generate(statistics, distribution, drift_limit):
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
