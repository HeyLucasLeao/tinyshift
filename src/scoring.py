import numpy as np
from scipy.stats import norm
from sklearn.metrics import f1_score
import pandas as pd


def metric_by_time_period(df, period, target_col, prediction_col, metric=f1_score):
    """
    Calculate a specified metric for each time period in the DataFrame.

    Parameters:
    - df: pandas DataFrame containing the data.
    - period: The time period for grouping data (e.g., 'W' for weeks, 'M' for months).
    - target_col: The name of the column containing the actual target values.
    - prediction_col: The name of the column containing the predicted values.
    - metric: A function to compute the metric (e.g., f1_score).

    Returns:
    - pandas DataFrame: A DataFrame with the metric calculated for each time period.
    """
    grouped = df.groupby(pd.Grouper(key="datetime", freq=period)).apply(
        lambda x: metric(x[target_col], x[prediction_col])
    )
    return grouped.reset_index(name="metric")


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


def reference_metrics_by_period(
    df,
    period,
    target_col,
    prediction_col,
    score_method,
    statistic,
    confidence_level=0.997,
    n_resamples=1000,
    random_state=42,
):
    """
    Calculate reference metrics for a DataFrame grouped by a specified
    time period, including confidence intervals and thresholds.

    Parameters:
    ----------
    - df : pd.DataFrame
        The input DataFrame containing the data.
    - period : str
        The time period for grouping (e.g., 'W' for weeks, 'M' for months).
    - target_col : str
        The column name containing the true labels or observed values.
    - prediction_col : str
        The column name containing the predicted values.
    - score_method : function
        The metric function to evaluate the model's performance (e.g., mean squared error, accuracy score).
    - statistic : function
        The statistical function to apply to the metrics (e.g., np.mean, np.median).
    - confidence_level : float, optional (default=0.997)
        The confidence level for the bootstrapped confidence interval (e.g., 0.95 for 95% confidence level).
    - n_resamples : int, optional (default=1000)
        The number of bootstrap resamples to perform for confidence interval calculation.
    - random_state : int, optional (default=42)
        Random seed for reproducibility.

    Returns:
    -------
    - dict
        A dictionary containing the following keys:
        - "ci_lower" : float
            The lower bound of the bias-corrected and accelerated (BCa) bootstrap confidence interval.
        - "ci_upper" : float
            The upper bound of the BCa bootstrap confidence interval.
        - "mean" : float
            The mean of the calculated metric across the specified period.
        - "lower_threshold" : float
            The lower threshold for anomaly detection, calculated as mean - 3 * standard deviation.
        - "upper_threshold" : float
            The upper threshold for anomaly detection, calculated as mean + 3 * standard deviation.

    Notes:
    ------
    - The confidence interval is calculated using the BCa two-sided bootstrap method.
    - The thresholds (lower and upper) are based on a 3-sigma rule assuming normal distribution.

    """
    # Calcular as métricas agrupadas por período
    metrics_by_period = metric_by_time_period(
        df, period, target_col, prediction_col, score_method
    )

    # Calcular o intervalo de confiança usando bootstrapping
    ci_lower, ci_upper = bootstrapping_bca(
        metrics_by_period["metric"],
        confidence_level,
        statistic,
        n_resamples,
        random_state,
    )

    # Calcular a média estimada da métrica
    estimated_mean_statistic = np.mean(metrics_by_period["metric"])

    # Calcular o desvio padrão
    std_deviation = metrics_by_period["metric"].std()

    # Calcular os limiares
    upper_threshold = estimated_mean_statistic + (std_deviation * 3)
    lower_threshold = estimated_mean_statistic - (std_deviation * 3)

    # Criar o dicionário de resultados
    results = {
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "mean": estimated_mean_statistic,
        "lower_threshold": lower_threshold,
        "upper_threshold": upper_threshold,
    }

    return results
