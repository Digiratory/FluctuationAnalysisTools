import os

import numpy as np
import pytest
from scipy import stats

from StatTools.analysis.dpcca import dpcca
from StatTools.generators.lbfbm_generator import LBFBmGenerator, normalize

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
if IN_GITHUB_ACTIONS:
    h_list = [0.5, 1, 1.5]
    rate_list = [14]
else:
    h_list = np.arange(0.5, 1.6, 0.15)
    rate_list = [12, 14]

testdata = {
    "h_list": h_list,
    "base_list": [1.1],
    "rate_list": rate_list,
}

STEP = 1


def calculate_hurst_exponent(
    trajectory: np.ndarray, scales: np.ndarray, step: float
) -> float:
    """
    Calculates the Hurst exponent from a given trajectory using the DPCCA algorithm.

    Args:
        trajectory (np.ndarray): The input trajectory to calculate the Hurst exponent for.
        scales (np.ndarray): An array of scales to use in the calculation.
        step (float): The step size used in the DPCCA algorithm.

    Returns:
        float: The calculated Hurst exponent.

    Notes:
        This function uses the dpcca module from StatTools.analysis.dpcca to perform the
        calculations. It normalizes the input trajectory, applies the DPCCA algorithm,
        and then calculates the Hurst exponent using linear regression.
    """
    signal_z = normalize(trajectory)
    _, _, f_z, s_z = dpcca(signal_z, 2, step, scales, processes=1, n_integral=1)
    f_z = np.sqrt(f_z)
    f_z /= f_z[0]
    res = stats.linregress(np.log(s_z), np.log(f_z)).slope
    return res


def get_test_h(
    base: float,
    h: float,
    scales: np.ndarray,
    step: int,
    target_len: int,
) -> float:
    """
    Calculates the Hurst exponent for the generated trajectory.

    Parameters:
        base: The base of the number system for bins
        filter_len: Filter length
        h: The specified Hurst exponent
        scales: Scales for analysis
        step: The step for analysis

    Returns:
        Calculated Hurst exponent (h_gen)
    """
    generator = LBFBmGenerator(h, base, length=target_len)
    trajectory = list(generator)
    res = calculate_hurst_exponent(trajectory, scales, step)
    return res


@pytest.mark.parametrize("h", testdata["h_list"])
@pytest.mark.parametrize("base", testdata["base_list"])
@pytest.mark.parametrize("rate", testdata["rate_list"])
def test_lbfbm_generator(h: float, base: float, rate: int):
    """
    It tests the generator for compliance with the specified Hurst exponent.

    Parameters:
        h: The specified Hurst exponent
        base: The base of the number system for bins
    """
    threshold = 0.10
    times = 3
    mean_difference = 0
    length = 2**rate
    scales = np.array([2**i for i in range(3, rate - 3)])
    for _ in range(times):
        h_gen = get_test_h(base, h, scales, STEP, length)

        mean_difference += abs(h_gen - h) / h
    mean_difference /= times
    assert (
        mean_difference <= threshold
    ), f"Diff between h and h_gen exceeds {threshold * 100}%: h={h}, h_gen={h_gen}, mean diff={mean_difference * 100:.2f}%"
