import os

import numpy as np
import pytest
from scipy import signal

from StatTools.analysis.dfa import DFA

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
TEST_H_VALUES_FULL = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
TEST_H_VALUES_CI = [0.5, 1.0, 1.5]  # Reduced set for CI environments

# Use reduced parameter set in GitHub Actions to speed up tests
TEST_H_VALUES = TEST_H_VALUES_CI if IN_GITHUB_ACTIONS else TEST_H_VALUES_FULL


def generate_fgn(length, h):
    """Generate fractional Gaussian noise with given Hurst exponent"""
    z = np.random.normal(size=length * 2)
    beta = 2 * h - 1
    L = length * 2
    A = np.zeros(length * 2)
    A[0] = 1
    for k in range(1, L):
        A[k] = (k - 1 - beta / 2) * A[k - 1] / k

    if h == 0.5:
        return z
    return signal.lfilter(1, A, z)


@pytest.mark.timeout(300)  # 5 minutes timeout
@pytest.mark.parametrize("h_target", TEST_H_VALUES)
def test_find_h_with_known_h(h_target):
    """Test DFA with signals of known Hurst exponent"""
    np.random.seed(42)
    length = 2**13
    data = generate_fgn(length, h_target)
    dfa = DFA(data)
    h = dfa.find_h()
    assert abs(h - h_target) < 0.05, f"Wrong h: expected {h_target}, got {h}"


def test_find_h_2d():
    """Test DFA with multiple time series"""
    np.random.seed(42)
    length = 2**13
    data = np.array([generate_fgn(length, h) for h in TEST_H_VALUES])
    dfa = DFA(data)
    h_values = dfa.find_h()

    assert isinstance(h_values, np.ndarray)
    assert h_values.shape == (len(TEST_H_VALUES),)
    for h_est, h_target in zip(h_values, TEST_H_VALUES):
        assert abs(h_est - h_target) < 0.1, f"Wrong h: expected {h_target}, got {h_est}"


def test_find_h_empty_input():
    """Test with empty input"""
    with pytest.raises(NameError) as exc_info:
        DFA([])
    assert "Wrong input array ! (It's probably too short)" in str(exc_info.value)


def test_find_h_short_input():
    """Test with too short input"""
    data = generate_fgn(1, 1.0)
    with pytest.raises(NameError) as exc_info:
        DFA(data)
    assert "Wrong input array ! (It's probably too short)" in str(exc_info.value)


def test_find_h_3d_input():
    """Test with 3D input (should raise error)"""
    data = np.random.normal(0, 1, (5, 5, 5))
    with pytest.raises(NameError) as exc_info:
        DFA(data)
    assert "Only 1- or 2-dimensional arrays are allowed!" in str(exc_info.value)


# TODO: fix tests for root and degree
# def test_find_h_with_root():
#     """Test with root=True option"""
#     np.random.seed(42)
#     length = 2**13
#     h_target = 0.8
#     data = generate_fgn(length, h_target)
#     dfa = DFA(data, root=True)
#     h = dfa.find_h()

#     assert isinstance(h, float)
#     assert abs(h - h_target) < 0.5, \
#         f"Wrong h: expected {h_target}, got {h}"

# def test_find_h_with_different_degree():
#     """Test with different polynomial degree"""
#     np.random.seed(42)
#     length = 2**13
#     h_target = 1.5
#     data = generate_fgn(length, h_target)
#     dfa = DFA(data, degree=3)
#     h = dfa.find_h()

#     assert isinstance(h, float)
#     assert abs(h - h_target) < 0.05, \
#         f"Wrong h: expected {h_target}, got {h}"
