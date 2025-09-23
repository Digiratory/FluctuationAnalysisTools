import numpy as np

from StatTools.analysis.dfa import DFA


def get_extra_h_dfa(signal):
    """
    Estimates Hurst exponent using DFA with extended range capability.

    For signals with extreme persistence properties (H < 0.5 or H > 1.5),
    this function applies signal transformations to bring the Hurst exponent
    into the measurable range and adjusts the final estimate accordingly.

    Parameters:
        signal (array_like): Input time series signal for which to estimate the Hurst exponent.

    Returns:
        float: Adjusted Hurst exponent estimate.
    """
    h = DFA(signal).find_h()
    new_h = h
    new_sihnal = signal
    if h > 1.5:
        # differentiate the signal
        diff_count = 0
        while new_h > 1.5:
            diff_count += 1
            new_sihnal = np.diff(new_sihnal)
            new_h = DFA(new_sihnal).find_h()
        new_h += diff_count

    elif h < 0.5:
        # integrate the signal
        integrate_count = 0
        while new_h < 0.5:
            integrate_count += 1
            new_sihnal = np.cumsum(new_sihnal)
            new_h = DFA(new_sihnal).find_h()
        new_h -= integrate_count
    return new_h
