"""Tests for the ndfnoise_generator module."""

import os

import numpy as np
import pytest

from StatTools.analysis.dfa import DFA
from StatTools.generators.ndfnoise_generator import ndfnoise

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
testdata = {
    "h_list_2d": [1, 1.2, 1.5],
    "h_list_3d": [1.1, 1.2, 1.5],
    "rate_2d": [10],
    "rate_3d": [9],
}


def get_h_dfa_sliced(arr: np.ndarray) -> np.ndarray:
    """
    Calculate DFA for z slices.
    """
    if len(arr.shape) != 3:
        h_values = DFA(arr).find_h()
        return h_values
    nx = arr.shape[0]
    nz = arr.shape[2]
    results = np.zeros(nx)

    for i in range(nz):
        if i % 250 != 0:
            continue
        slice_2d = arr[:, :, i]
        h_values = DFA(slice_2d).find_h()
        results[i] = np.mean(h_values)
    return results


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test too long for Github Actions.")
@pytest.mark.parametrize("hurst_theory", testdata["h_list_2d"])
@pytest.mark.parametrize("rate", testdata["rate_2d"])
def test_ndfnoise_generator_2d(hurst_theory: float, rate: int):
    """Generator test"""
    size = 2**rate
    dim = 2
    shape = (size,) * dim
    f = ndfnoise(shape, hurst_theory, normalize=True)
    hurst_est_array = get_h_dfa_sliced(np.diff(f))
    hurst_mean = np.mean(hurst_est_array)
    assert np.isclose(
        hurst_mean, hurst_theory, atol=0.7
    ).all(), f"Hurst mismatch: estimated={hurst_mean}, expected={hurst_theory}"


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test too long for Github Actions.")
@pytest.mark.parametrize("hurst_theory", testdata["h_list_3d"])
@pytest.mark.parametrize("rate", testdata["rate_3d"])
def test_ndfnoise_generator_3d(hurst_theory: float, rate: int):
    """Generator test"""
    size = 2**rate
    dim = 3
    shape = (size,) * dim
    f = ndfnoise(shape, hurst_theory, normalize=True)
    hurst_est_array = get_h_dfa_sliced(np.diff(f))
    hurst_mean = np.mean(hurst_est_array, where=hurst_est_array != 0)
    assert np.isclose(
        hurst_mean, hurst_theory, atol=0.2
    ).all(), f"Hurst mismatch: estimated={hurst_mean}, expected={hurst_theory}"
