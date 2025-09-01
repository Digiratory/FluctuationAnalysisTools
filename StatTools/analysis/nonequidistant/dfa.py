import itertools
from multiprocessing import Pool

import numpy as np
import pandas as pd


def __dfa_polyfit_core(windowed: pd.Series, order: int) -> list:
    """
    Core detrending function for DFA analysis on a single time window.

    Fits a polynomial of specified order to the data in a time window and
    returns the detrended residuals. This function performs the local
    detrending step of the DFA algorithm.

    Args:
        windowed (pd.Series): Time series data within a single analysis window
        order (int): Order of polynomial for detrending (typically 1 or 2)

    Returns:
        list: Detrended residuals (fluctuations) for the window. Returns empty
            list if window contains insufficient data points for polynomial fitting.
    """
    # Drop windows that contains less than needed for approximation
    # TODO: probably better to drop windows that contains less than some amount of non perfectly fitted points?
    if len(windowed.index) < order + 1:
        return []
    times = windowed.index.map(lambda x: x.value).to_numpy()
    z = np.polyfit(times, windowed.to_numpy(), order)
    current_trend = np.polyval(z, times)
    detrended_window = windowed - current_trend
    return list(detrended_window.values)


def __dfa_scale_core(
    dataset: pd.Series,
    scale: pd.Timedelta,
    step: pd.Timedelta,
    order: int,
    num_process=None,
) -> float:
    """
    Compute DFA fluctuation function for a specific scale on non-equidistant data.

    Divides the time series into overlapping windows of specified scale and
    computes the root-mean-square fluctuation after detrending each window.
    Supports parallel processing for improved performance on large datasets.

    Args:
        dataset (pd.Series): Input time series with datetime index
        scale (pd.Timedelta): Time scale for analysis windows
        step (pd.Timedelta): Step size between consecutive windows (None for no overlap)
        order (int): Polynomial order for detrending
        num_process (int): Number of parallel processes (None for sequential processing)

    Returns:
        float: Root-mean-square fluctuation F(s) for the given scale, or np.nan
            if no valid windows are found
    """
    residuals = []
    start_time = dataset.index[0]
    start_times = []

    if step is None:
        start_times = list(dataset[: dataset.index[-1] - scale].index)
    else:
        while start_time < dataset.index[-1] - scale:
            start_time += step
            start_times.append(start_time)

    if len(start_times) == 0:
        print(f"No valid windows with scale {scale} found.")
        return np.nan

    if num_process is None:
        for start_time in start_times:
            residuals.extend(
                __dfa_polyfit_core(dataset[start_time : start_time + scale], order)
            )
    else:
        assert num_process > 0, "Invalid value for 'process'"
        num_process = min(num_process, len(start_times))
        with Pool(num_process) as p:
            residuals = list(
                itertools.chain(
                    *list(
                        p.starmap(
                            __dfa_polyfit_core,
                            [
                                (dataset[start_time : start_time + scale].copy(), order)
                                for start_time in start_times
                            ],
                        )
                    )
                )
            )
    residuals = np.array(residuals)
    F_2 = np.mean(residuals**2)
    return F_2


def dfa(
    data: pd.Series,
    scales: list[pd.Timedelta],
    step: float = None,
    order: int = 2,
    num_process=None,
) -> pd.Series:
    """
    Calculate the Detrended Fluctuation Analysis (DFA) for non-equidistant time series.

    This function extends traditional DFA to handle time series with irregular sampling
    intervals (non-equidistant data). It computes the fluctuation function F(s) at
    different time scales s, which can be used to estimate the Hurst exponent and
    detect long-range correlations.

    The algorithm works by:
    1. Dividing the time series into windows of size s
    2. Fitting polynomials to each window to remove local trends
    3. Computing RMS fluctuations of the detrended data
    4. Analyzing scaling behavior F(s) ~ s^H

    Basic usage:
        ```python
        import pandas as pd
        import numpy as np
        from StatTools.analysis.nonequidistant.dfa import dfa

        # Create sample non-equidistant time series
        times = pd.date_range('2020-01-01', periods=1000, freq='1H')
        # Remove some points to make it non-equidistant
        times = times[np.random.choice(len(times), size=800, replace=False)]
        values = np.random.normal(0, 1, len(times))
        ts = pd.Series(values, index=times)

        # Define scales for analysis
        scales = [pd.Timedelta(hours=h) for h in [1, 2, 4, 8, 16, 32]]

        # Perform DFA
        fluctuations = dfa(ts, scales, step=0.5, order=2)
        ```

    Args:
        data (pd.Series): Input time series data. Must have a DatetimeIndex.
            Contains absolute measurements (not increments) as typically used
            in fluctuation analysis.
        scales (list[pd.Timedelta]): List of time scales for analysis.
            Each scale defines the window size for detrending.
        step (float, optional): Step fraction for window overlap (0 < step <= 1).
            If None, windows don't overlap. Default creates 50% overlap.
        order (int, optional): Polynomial order for detrending (default: 2).
            Higher orders remove more complex trends.
        num_process (int, optional): Number of parallel processes for computation.
            If None, processing is sequential. Improves performance for large datasets.

    Returns:
        pd.Series: Fluctuation function F²(s) values indexed by time scales.
            Values represent the squared RMS fluctuations at each scale.

    Raises:
        ValueError: If data index is not a DatetimeIndex or if parameters are invalid

    Warning:
        Current implementation may be unstable for Hurst exponents H < 1 when
        data contains significant gaps (~25% of equivalent equidistant data).

    Note:
        For best results, ensure adequate data coverage within each time scale window.
        Very sparse data at large scales may lead to unreliable fluctuation estimates.
    """
    # Ensure the index is a valid timeseries
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("The index of the Series must be a valid timeseries")

    FF_2 = []

    for scale in scales:
        if step is not None:
            step_scale = step * scale
        else:
            step_scale = None
        FF_2.append(__dfa_scale_core(data, scale, step_scale, order, num_process))

    return pd.Series(FF_2, index=scales)
