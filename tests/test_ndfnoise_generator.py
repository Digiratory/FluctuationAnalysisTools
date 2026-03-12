"""Tests for the ndfnoise_generator module."""

import os
from multiprocessing import Pool

import numpy as np
import pytest

from StatTools.analysis import analyse_zero_cross_ff
from StatTools.analysis.dfa import dfa
from StatTools.generators.ndfnoise_generator import ndfnoise

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
testdata = {
    "h_list_2d": [0.55, 0.8, 1.2, 1.5],
    "h_list_3d": [1.1, 1.2, 1.5],
    "rate_2d": [9],
    "rate_3d": [9],
}


def _process_2d_slice(slice_2d: np.ndarray) -> float:
    """
    Calculate DFA for slice.
    """
    s_vals, f2_vals = dfa(slice_2d, degree=2, processes=1)
    hs = np.sqrt(f2_vals)
    ff_parameters, _ = analyse_zero_cross_ff(hs, s_vals)
    return ff_parameters.slopes[0].value


def get_h_dfa_sliced(arr: np.ndarray) -> float:
    """
    Calculate DFA for z slices.
    For 2D: single Hurst exponent.
    For 3D: array of Hurst exponents for each 250th slice.
    """
    if len(arr.shape) == 2:
        return _process_2d_slice(arr)
    nz = arr.shape[2]
    slice_indices = np.arange(0, nz, 250)

    with Pool() as pool:
        slices = [arr[:, :, i] for i in slice_indices]
        results = pool.map(_process_2d_slice, slices)

    return np.array(results)


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
        hurst_mean, hurst_theory, atol=0.2
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
