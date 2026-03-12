import os

import numpy as np
import pytest

from StatTools.analysis.emd_mfdfa import emd_dfa, emd_mfdfa
from StatTools.generators.kasdin_generator import create_kasdin_generator

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
TEST_H_VALUES_FULL = [0.3, 0.5, 0.7]
TEST_H_VALUES_CI = [0.5, 0.7]

TEST_H_VALUES = TEST_H_VALUES_CI if IN_GITHUB_ACTIONS else TEST_H_VALUES_FULL


@pytest.mark.timeout(300)
@pytest.mark.parametrize("h_target", TEST_H_VALUES)
def test_emd_mfdfa_estimate_hurst(h_target):
    """Test that EMD-MFDFA estimates the Hurst exponent within tolerance."""
    np.random.seed(42)
    length = 2**13  # 8192
    generator = create_kasdin_generator(h=h_target, length=length, normalize=True)
    signal = generator.get_full_sequence()

    q_vals, h_q, _ = emd_mfdfa(signal)
    q2_idx = np.argmin(np.abs(q_vals - 2.0))
    h_estimated = h_q[q2_idx]

    assert (
        abs(h_estimated - h_target) < 0.15
    ), f"Wrong h: expected {h_target}, got {h_estimated}"


def test_emd_mfdfa_full_analysis():
    """Test that the function returns correct types and shapes."""
    np.random.seed(42)
    length = 1024
    generator = create_kasdin_generator(h=0.6, length=length, normalize=True)
    signal = generator.get_full_sequence()

    q_values, h_q, scales = emd_mfdfa(signal)

    assert isinstance(q_values, np.ndarray)
    assert isinstance(h_q, np.ndarray)
    assert isinstance(scales, np.ndarray)

    assert len(q_values) == len(h_q)
    assert len(scales) > 0


def test_emd_mfdfa_short_input():
    """Test with too short input."""
    signal = np.random.normal(0, 1, 32)
    with pytest.raises(ValueError) as exc_info:
        emd_mfdfa(signal)
    assert "Signal too short" in str(exc_info.value)


def test_emd_mfdfa_dimensions():
    """Test with 2D input (should raise error)."""
    data = np.random.normal(0, 1, (100, 2))
    with pytest.raises(ValueError) as exc_info:
        emd_mfdfa(data)
    assert "must be a 1D array" in str(exc_info.value)


def test_emd_mfdfa_custom_q():
    """Test with custom q values, especially handling q=0."""
    np.random.seed(42)
    signal = np.random.normal(0, 1, 1024)
    custom_q = np.array([-2, 0, 2])
    q_vals, h_q, _ = emd_mfdfa(signal, q_values=custom_q)
    assert np.array_equal(q_vals, custom_q)
    assert len(h_q) == 3
    assert not np.isnan(h_q).any()


def test_emd_mfdfa_length_warning():
    """Test warning for short input length."""
    signal = np.random.normal(0, 1, 256)
    with pytest.warns(UserWarning, match="too short for reliable"):
        emd_mfdfa(signal)


@pytest.mark.timeout(300)
@pytest.mark.parametrize("h_target", TEST_H_VALUES)
def test_emd_dfa_estimate_hurst(h_target):
    """Test that EMD-DFA H estimate (from log-log slope) is within tolerance."""
    np.random.seed(42)
    length = 2**13
    generator = create_kasdin_generator(h=h_target, length=length, normalize=True)
    signal = generator.get_full_sequence()

    scales, F2_s = emd_dfa(signal)

    valid = F2_s > 0
    coeffs = np.polyfit(np.log(scales[valid]), np.log(F2_s[valid]), 1)
    h_estimated = coeffs[0] / 2.0

    assert (
        abs(h_estimated - h_target) < 0.15
    ), f"Wrong h: expected {h_target}, got {h_estimated:.3f}"


def test_emd_dfa_returns_types_and_shapes():
    """Test that emd_dfa returns correct types and non-empty arrays."""
    np.random.seed(42)
    length = 1024
    generator = create_kasdin_generator(h=0.6, length=length, normalize=True)
    signal = generator.get_full_sequence()

    scales, F2_s = emd_dfa(signal)

    assert isinstance(scales, np.ndarray)
    assert isinstance(F2_s, np.ndarray)
    assert len(scales) == len(F2_s)
    assert len(scales) > 0
    assert (F2_s > 0).all()


def test_emd_dfa_short_input():
    """Test that emd_dfa raises ValueError for too-short signals."""
    signal = np.random.normal(0, 1, 32)
    with pytest.raises(ValueError, match="Signal too short"):
        emd_dfa(signal)


def test_emd_dfa_2d_input():
    """Test that emd_dfa raises ValueError for 2D input."""
    data = np.random.normal(0, 1, (100, 2))
    with pytest.raises(ValueError, match="must be a 1D array"):
        emd_dfa(data)


def test_emd_dfa_length_warning():
    """Test that emd_dfa warns for short-but-valid signals."""
    signal = np.random.normal(0, 1, 256)
    with pytest.warns(UserWarning, match="too short for reliable"):
        emd_dfa(signal)


def test_emd_dfa_custom_scales():
    """Test that emd_dfa respects user-supplied scales."""
    np.random.seed(42)
    signal = np.random.normal(0, 1, 2048)
    custom_scales = np.array([32, 64, 128, 256])

    scales, F2_s = emd_dfa(signal, scales=custom_scales)

    assert len(scales) <= len(custom_scales)
    assert set(scales).issubset(set(custom_scales))


def test_emd_dfa_accepts_list_input():
    """Test that emd_dfa works with a plain Python list."""
    np.random.seed(42)
    signal_list = np.random.normal(0, 1, 1024).tolist()

    scales, F2_s = emd_dfa(signal_list)

    assert isinstance(scales, np.ndarray)
    assert len(scales) > 0


def test_emd_dfa_positive_scaling():
    """log-log slope of F²(s) should be positive for persistent fGn (H>0.5)."""
    np.random.seed(42)
    generator = create_kasdin_generator(h=0.7, length=2**13, normalize=True)
    signal = generator.get_full_sequence()

    scales, F2_s = emd_dfa(signal)

    slope = np.polyfit(np.log(scales), np.log(F2_s), 1)[0]
    assert slope > 0, f"Expected positive log-log slope, got {slope:.3f}"
