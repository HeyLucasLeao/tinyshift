from typing import Callable, Tuple, Union
import numpy as np


class StatisticalInterval:
    """Utility class to compute different types of statistical interval."""

    @staticmethod
    def custom_interval(data: np.ndarray, custom_func: Callable) -> Tuple[float, float]:
        """Calculate interval using a custom function."""
        return custom_func(data)

    @staticmethod
    def calculate_interval(
        X: np.ndarray,
        center: Callable,
        spread: Callable,
        factor: float = 3,
    ) -> Tuple[float, float]:
        """Calculate interval using a central tendency and spread function."""

        if not callable(center) or not callable(spread):
            raise ValueError("center and spread must be callable functions")

        center_value = center(X)
        spread_value = spread(X)
        lower_bound = center_value - factor * spread_value
        upper_bound = center_value + factor * spread_value
        return lower_bound, upper_bound

    @staticmethod
    def iqr_interval(X: np.ndarray) -> Tuple[float, float]:
        """Calculates interval using IQR and median with a default factor of 1.5."""

        def iqr(x):
            q75, q25 = np.percentile(x, [75, 25])
            return q75 - q25

        return StatisticalInterval.calculate_interval(X, np.median, iqr, factor=1.5)

    @staticmethod
    def stddev_interval(X: np.ndarray) -> Tuple[float, float]:
        """Calculates interval using mean and standard deviation."""
        return StatisticalInterval.calculate_interval(X, np.mean, np.std)

    @staticmethod
    def mad_interval(X: np.ndarray) -> Tuple[float, float]:
        """Calculates interval using Median Absolute Deviation (MAD)."""
        mad = lambda x: np.median(np.abs(x - np.median(x)))
        return StatisticalInterval.calculate_interval(X, np.median, mad)

    @staticmethod
    def quantile_interval(
        X: np.ndarray, lower: float, upper: float
    ) -> Tuple[float, float]:
        """Calculates interval using quantiles."""
        lower_bound = (
            np.quantile(X, lower, method="higher") if lower is not None else None
        )
        upper_bound = (
            np.quantile(X, upper, method="higher") if upper is not None else None
        )
        return (lower_bound, upper_bound)

    @staticmethod
    def compute_interval(
        X: np.ndarray,
        method: Union[str, Callable, Tuple[float]],
    ) -> Tuple[float, float]:
        """
        Determines the lower and upper bounds on the specified method.

        Args:
            data: Input data for threshold calculation.
            method: Method to compute interval. Can be:
                - "stddev" (mean ± 3σ)
                - "mad" (median ± 3*MAD)
                - "iqr" (median ± 1.5*IQR)
                - A custom function (returns lower, upper bounds)
                - A pre-defined tuple (lower, upper)

        Returns:
            Tuple[float, float]: Lower and upper bounds.
        """
        X = np.asarray(X)
        if isinstance(method, str):
            if method == "stddev":
                lower_bound, upper_bound = StatisticalInterval.stddev_interval(X)
            elif method == "mad":
                lower_bound, upper_bound = StatisticalInterval.mad_interval(X)
            elif method == "iqr":
                lower_bound, upper_bound = StatisticalInterval.iqr_interval(X)
            else:
                raise ValueError(f"Unsupported method: {method}")
        elif callable(method):
            lower_bound, upper_bound = StatisticalInterval.custom_interval(X, method)
        elif isinstance(method, tuple) and len(method) == 2:
            lower_bound, upper_bound = method
        elif isinstance(method, tuple) and len(method) == 3 and method[0] == "quantile":
            lower, upper = method[1], method[2]
            lower_bound, upper_bound = StatisticalInterval.quantile_interval(
                X, lower, upper
            )
        else:
            raise ValueError("Invalid method specification.")

        return lower_bound, upper_bound
