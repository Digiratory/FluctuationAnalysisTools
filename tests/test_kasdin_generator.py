import os

import numpy as np
import pytest

from StatTools.experimental.analysis.tools import get_extra_h_dfa
from StatTools.generators.kasdin_generator import create_kasdin_generator

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
if IN_GITHUB_ACTIONS:
    h_list = [0.25, 1, 2, 3.5]
else:
    h_list = np.arange(0.25, 3.75, 0.25)

testdata = {
    "h_list": h_list,
    "rate_list": [14],
}


def get_test_h(h: float, target_len: int, filter_coefficients_length: int) -> float:
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
    generator = create_kasdin_generator(
        h, length=target_len, filter_coefficients_length=filter_coefficients_length
    )
    signal = generator.get_full_sequence()
    return get_extra_h_dfa(signal)


@pytest.mark.parametrize("h", testdata["h_list"])
@pytest.mark.parametrize("rate", testdata["rate_list"])
def test_kasdin_generator_open_length(h: float, rate: int):
    """
    It tests the generator for compliance with the specified Hurst exponent.

    Parameters:
        h: The specified Hurst exponent
        base: The base of the number system for bins
    """
    threshold = 0.20
    times = 5
    mean_difference = 0
    length = 2**rate
    for _ in range(times):
        h_gen = get_test_h(h, length, length)
        mean_difference += abs(h_gen - h) / h
    mean_difference /= times
    assert (
        mean_difference <= threshold
    ), f"Diff between h and h_gen exceeds {threshold * 100}%: h={h}, h_gen={h_gen}, mean diff={mean_difference * 100:.2f}%"
