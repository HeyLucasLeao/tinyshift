import numpy as np
from typing import List, Union


def hpi(
    y_hat: Union[np.ndarray, List[float]],
    w_s: float,
) -> np.ndarray:
    """
    Horizontal Partial Interpolation (HPI) combines the original forecast of the current horizon with the
    original forecast of the previous horizon using a weight value. This technique helps stabilize
    forecasts by reducing the variability between consecutive forecast horizons.

    Parameters
    ----------
    y_hat : array-like
        List or array of original forecasts [F_H1, F_H2, F_H3, ...] representing forecasts for different horizons.
        Must be one-dimensional.
    w_s : float
        Weight for the previous horizon forecast (0 <= w_s <= 1). Higher values give more weight to
        the previous horizon, resulting in smoother forecasts.

    Returns
    -------
    numpy.ndarray
        Array of stable forecasts [SF_H1, SF_H2, SF_H3, ...] where each element is the interpolated
        forecast for the corresponding horizon.

    Raises
    ------
    ValueError
        If the input forecast series is not one-dimensional.

    Notes
    -----
    - The first forecast (SF_H1) equals the original forecast (F_H1)
    - Higher w_s values create smoother, more stable forecast trajectories
    - Lower w_s values preserve more of the original forecast dynamics

    References
    ----------
    - Godahewa, R., Bergmeir, C., Erkin Baz, Z., Zhu, C., Song, Z., García, S.,
      & Benavides, D. (2023). On forecast stability. International Journal of
      Forecasting, 41(4), 1539-1558.
    """

    y_hat = np.asarray(y_hat)

    if y_hat.ndim != 1:
        raise ValueError("The original forecast series must be one-dimensional.")

    fc = y_hat.copy()

    for i in range(1, fc.shape[0]):
        fc[i] = (w_s * y_hat[i - 1]) + ((1 - w_s) * y_hat[i])
    return fc


def hfi(
    y_hat: Union[np.ndarray, List[float]],
    w_s: float,
) -> np.ndarray:
    """
    Horizontal Full Interpolation (HFI) is a forecast combination technique that blends
    the stable forecast from the previous horizon with the original forecast of the current
    horizon using a weight value. This technique helps stabilize forecasts by reducing the variability between consecutive forecast horizons.

    Parameters
    ----------
    y_hat : array-like
        List or array of original forecasts [F_H1, F_H2, F_H3, ...] from a single origin.
        Must be one-dimensional.
    w_s : float
        Weight for the previous stable forecast (0 <= w_s <= 1).
        Higher values give more weight to the previous stable forecast, creating smoother trajectories.

    -------
    numpy.ndarray
        Array of stable forecasts [SF_H1, SF_H2, SF_H3, ...] after applying HFI.
        The first element remains unchanged (SF_H1 = F_H1).

    Raises
    ------
    ValueError
        If the input forecast series is not one-dimensional.

    Notes
    -----
    - The first forecast (SF_H1) equals the original forecast (F_H1)
    - Higher w_s values create smoother, more stable forecast trajectories
    - Lower w_s values preserve more of the original forecast dynamics

    References
    ----------
    - Godahewa, R., Bergmeir, C., Erkin Baz, Z., Zhu, C., Song, Z., García, S.,
      & Benavides, D. (2023). On forecast stability. International Journal of
      Forecasting, 41(4), 1539-1558.
    """
    y_hat = np.asarray(y_hat)

    if y_hat.ndim != 1:
        raise ValueError("The original forecast series must be one-dimensional.")

    fc = y_hat.copy()

    for i in range(1, fc.shape[0]):
        fc[i] = (w_s * fc[i - 1]) + ((1 - w_s) * fc[i])

    return fc
