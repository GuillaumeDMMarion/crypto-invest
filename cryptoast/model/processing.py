"""Data processing.
"""
import numpy as np
from scipy import interpolate
from scipy.stats import genextreme


def forecast_params(
    data,
    ticks=(0.7, 0.8, 0.9, 1),
    interpolator=interpolate.InterpolatedUnivariateSpline,
    **kwargs
):
    """Forecast rvs params."""
    all_params = [genextreme.fit(data[: int(_ * data.size)]) for _ in ticks]
    args = [params[:-2] for params in all_params]
    locs = [params[-2] for params in all_params]
    scales = [params[-1] for params in all_params]
    arg, loc, scale = [
        interpolator(range(len(_)), _, **kwargs)(len(_) + 1)
        for _ in (args, locs, scales)
    ]
    arg = args[-1]
    scale = scales[-1]
    return arg, loc, scale


def get_rvs_data(
    data,
    validation_window,
    ticks=(0.7, 0.8, 0.9, 1),
    interpolator=interpolate.InterpolatedUnivariateSpline,
    **kwargs
):
    """Get distribution data."""
    arg, loc, scale = forecast_params(
        data=data, ticks=ticks, interpolator=interpolator, **kwargs
    )
    scale = max(0, scale)
    rvs_data = genextreme.rvs(c=arg, loc=loc, scale=scale, size=validation_window)
    return rvs_data


def corrector(data, validation_window):
    """Correct data."""
    rvs_data = get_rvs_data(data, validation_window * 2).reshape(-1, 1)
    length = int(rvs_data.shape[0])
    factor = 0
    return np.vstack([data[-length * factor :], rvs_data])  # rvs_data
