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
        data: np.ndarray,
        center: Callable,
        spread: Callable,
        factor: float = 3,
    ) -> Tuple[float, float]:
        """Calculate interval using a central tendency and spread function."""

        if not callable(center) or not callable(spread):
            raise ValueError("center and spread must be callable functions")

        center_value = center(data)
        spread_value = spread(data)
        lower_bound = center_value - factor * spread_value
        upper_bound = center_value + factor * spread_value
        return lower_bound, upper_bound

    @staticmethod
    def iqr_interval(data: np.ndarray) -> Tuple[float, float]:
        """Calculates interval using IQR and median with a default factor of 1.5."""

        def iqr(x):
            q75, q25 = np.percentile(x, [75, 25])
            return q75 - q25

        return StatisticalInterval.calculate_interval(data, np.median, iqr, factor=1.5)

    @staticmethod
    def stddev_interval(data: np.ndarray) -> Tuple[float, float]:
        """Calculates interval using mean and standard deviation."""
        return StatisticalInterval.calculate_interval(data, np.mean, np.std)

    @staticmethod
    def mad_interval(data: np.ndarray) -> Tuple[float, float]:
        """Calculates interval using Median Absolute Deviation (MAD)."""
        mad = lambda x: np.median(np.abs(x - np.median(x)))
        return StatisticalInterval.calculate_interval(data, np.median, mad)

    @staticmethod
    def compute_interval(
        data: np.ndarray,
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
        data = np.asarray(data)

        if isinstance(method, str):
            if method == "stddev":
                lower_bound, upper_bound = StatisticalInterval.stddev_interval(data)
            elif method == "mad":
                lower_bound, upper_bound = StatisticalInterval.mad_interval(data)
            elif method == "iqr":
                lower_bound, upper_bound = StatisticalInterval.iqr_interval(data)
            else:
                raise ValueError(f"Unsupported method: {method}")
        elif callable(method):
            lower_bound, upper_bound = StatisticalInterval.custom_interval(data, method)
        elif isinstance(method, tuple) and len(method) == 2:
            lower_bound, upper_bound = method
        else:
            raise ValueError("Invalid method specification.")

        return lower_bound, upper_bound
