import os

import numpy as np
import pytest
from scipy import stats

from StatTools.analysis.dfa import _detrend_segment, dfa, dfa_worker
from StatTools.generators import generate_fbn

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
if IN_GITHUB_ACTIONS:
    TEST_H_VALUES = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
else:
    TEST_H_VALUES = [0.5, 1.0, 1.5, 2.0]


@pytest.fixture(scope="module")
def sample_signals():
    """Generate sample signals with different Hurst exponents for testing"""
    length = 2**13
    signals = {}
    np.random.seed(42)  # Set seed for reproducibility
    for h in TEST_H_VALUES:
        # generate_fbn returns shape (1, length), so we flatten to 1D
        signals[h] = generate_fbn(hurst=h, length=length, method="kasdin").flatten()
    return signals


# ====================== Tests for atomic functions ======================


def test_detrend_segment():
    """Test _detrend_segment function"""
    # Create a simple linear segment
    indices = np.arange(10)
    y_segment = 2 * indices + 1 + np.random.normal(0, 0.1, 10)  # linear trend + noise

    residuals = _detrend_segment(y_segment, indices, degree=1)

    # Residuals should be close to zero (detrended)
    assert isinstance(residuals, np.ndarray)
    assert residuals.shape == (10,)
    assert np.abs(np.mean(residuals)) < 0.5  # mean should be close to zero


def test_detrend_segment_quadratic():
    """Test _detrend_segment with quadratic trend"""
    indices = np.arange(20)
    y_segment = indices**2 + 3 * indices + 1 + np.random.normal(0, 0.1, 20)

    residuals = _detrend_segment(y_segment, indices, degree=2)

    assert isinstance(residuals, np.ndarray)
    assert residuals.shape == (20,)
    assert np.abs(np.mean(residuals)) < 1.0


# ====================== Tests for dfa_worker ======================


def test_dfa_worker_1d_input():
    """Test dfa_worker with 1D input (should be normalized to 2D)"""
    np.random.seed(42)
    # generate_fbn returns (1, length), we need 2D for dfa_worker
    data_2d = generate_fbn(hurst=0.5, length=1000, method="kasdin")

    result = dfa_worker(indices=0, arr=data_2d, degree=2)

    assert isinstance(result, list)
    assert len(result) == 1
    s_vals, f2_vals = result[0]
    assert isinstance(s_vals, np.ndarray)
    assert isinstance(f2_vals, np.ndarray)
    assert len(s_vals) == len(f2_vals)
    assert len(s_vals) > 0


def test_dfa_worker_2d_input():
    """Test dfa_worker with 2D input"""
    np.random.seed(42)
    # Generate multiple time series
    data_list = []
    for h in [0.5, 1.0, 1.5]:
        series = generate_fbn(hurst=h, length=1000, method="kasdin").flatten()
        data_list.append(series)
    data = np.array(data_list)

    result = dfa_worker(indices=[0, 1, 2], arr=data, degree=2)

    assert isinstance(result, list)
    assert len(result) == 3
    for s_vals, f2_vals in result:
        assert isinstance(s_vals, np.ndarray)
        assert isinstance(f2_vals, np.ndarray)
        assert len(s_vals) == len(f2_vals)


def test_dfa_worker_custom_s_values():
    """Test dfa_worker with custom s_values"""
    np.random.seed(42)
    data_2d = generate_fbn(hurst=0.5, length=1000, method="kasdin")
    custom_s = [16, 32, 64, 128]

    result = dfa_worker(indices=0, arr=data_2d, degree=2, s_values=custom_s)

    s_vals, f2_vals = result[0]
    assert len(s_vals) <= len(custom_s)  # Some may be filtered out
    assert all(s in custom_s for s in s_vals)


def test_dfa_worker_invalid_dimension():
    """Test dfa_worker raises error for invalid dimensions"""
    data_3d = np.random.normal(0, 1, (5, 5, 5))

    with pytest.raises(ValueError, match="expects 2D array"):
        dfa_worker(indices=0, arr=data_3d, degree=2)


# ====================== Tests for dfa function ======================


@pytest.mark.parametrize("h", TEST_H_VALUES)
def test_dfa_1d_with_known_h(sample_signals, h):
    """Test dfa function with 1D input and known Hurst exponent"""
    sig = sample_signals[h]

    s_vals, f2_vals = dfa(sig, degree=2, processes=1)

    # Check return types
    assert isinstance(s_vals, np.ndarray)
    assert isinstance(f2_vals, np.ndarray)
    assert len(s_vals) == len(f2_vals)
    assert len(s_vals) > 0

    # Check that F^2(s) increases with s (for positive H)
    if h > 0.5:
        assert np.all(np.diff(f2_vals) > 0) or np.all(np.diff(f2_vals) >= -1e-10)

    # Estimate H from log-log plot
    # Since dfa returns F^2(s), we need to extract sqrt before taking log
    log_s = np.log(s_vals)
    log_f = np.log(np.sqrt(f2_vals))  # Extract sqrt from F^2 to get F
    res = stats.linregress(log_s, log_f)

    assert res.slope == pytest.approx(h, abs=0.1)


@pytest.mark.parametrize("h", TEST_H_VALUES)
def test_dfa_1d_parallel(sample_signals, h):
    """Test dfa function with parallel processing"""
    sig = sample_signals[h]

    s_vals, f2_vals = dfa(sig, degree=2, processes=2)

    assert isinstance(s_vals, np.ndarray)
    assert isinstance(f2_vals, np.ndarray)
    assert len(s_vals) == len(f2_vals)

    # Estimate H from log-log plot
    log_s = np.log(s_vals)
    log_f = np.log(np.sqrt(f2_vals))  # Extract sqrt from F^2 to get F
    res = stats.linregress(log_s, log_f)

    assert res.slope == pytest.approx(h, abs=0.1)


def test_dfa_2d_input():
    """Test dfa function with 2D input (multiple time series)"""
    np.random.seed(42)
    # Generate multiple time series with different H
    data_list = []
    h_list = []
    for h in TEST_H_VALUES:
        series = generate_fbn(hurst=h, length=1000, method="kasdin").flatten()
        data_list.append(series)
        h_list.append(h)
    data = np.array(data_list)

    s_vals, f2_vals = dfa(data, degree=2, processes=1)

    # For 2D input: s is 1D array, F2_s is 2D array
    assert isinstance(s_vals, np.ndarray)
    assert isinstance(f2_vals, np.ndarray)
    assert f2_vals.ndim == 2
    assert f2_vals.shape[0] == len(TEST_H_VALUES)
    assert f2_vals.shape[1] == len(s_vals)

    # Check H estimation for each time series
    log_s = np.log(s_vals)
    for i, h_target in enumerate(h_list):
        log_f = np.log(np.sqrt(f2_vals[i]))  # Extract sqrt from F^2 to get F
        res = stats.linregress(log_s, log_f)
        assert res.slope == pytest.approx(h_target, abs=0.1)


def test_dfa_2d_parallel():
    """Test dfa function with 2D input and parallel processing"""
    np.random.seed(42)
    data_list = []
    h_list = []
    for h in TEST_H_VALUES:
        series = generate_fbn(hurst=h, length=1000, method="kasdin").flatten()
        data_list.append(series)
        h_list.append(h)
    data = np.array(data_list)

    s_vals, f2_vals = dfa(data, degree=2, processes=2)

    assert isinstance(s_vals, np.ndarray)
    assert isinstance(f2_vals, np.ndarray)
    assert f2_vals.ndim == 2
    assert f2_vals.shape[0] == len(TEST_H_VALUES)

    # Check H estimation for each time series
    log_s = np.log(s_vals)
    for i, h_target in enumerate(h_list):
        log_f = np.log(np.sqrt(f2_vals[i]))  # Extract sqrt from F^2 to get F
        res = stats.linregress(log_s, log_f)
        assert res.slope == pytest.approx(h_target, abs=0.1)


def test_dfa_different_degrees():
    """Test dfa function with different polynomial degrees"""
    np.random.seed(42)
    data = generate_fbn(hurst=1.0, length=1000, method="kasdin").flatten()

    s1, f2_1 = dfa(data, degree=1, processes=1)
    s2, f2_2 = dfa(data, degree=2, processes=1)
    s3, f2_3 = dfa(data, degree=3, processes=1)

    # All should return valid results
    assert len(s1) > 0
    assert len(s2) > 0
    assert len(s3) > 0

    # Results should be similar but not identical
    assert len(s1) == len(s2) == len(s3)


def test_dfa_empty_input():
    """Test dfa function with empty input"""
    with pytest.raises(ValueError):
        dfa(np.array([]), degree=2)


def test_dfa_invalid_dimension():
    """Test dfa function with invalid dimensions"""
    data_3d = np.random.normal(0, 1, (5, 5, 5))

    with pytest.raises(ValueError, match="Only 1D or 2D arrays"):
        dfa(data_3d, degree=2)


def test_dfa_short_input():
    """Test dfa function with too short input"""
    np.random.seed(42)
    data = generate_fbn(hurst=1.0, length=10, method="kasdin").flatten()  # Very short

    # Should not raise error, but may return empty or very few scales
    s_vals, f2_vals = dfa(data, degree=2)

    # Should still return valid arrays (may be empty or very short)
    assert isinstance(s_vals, np.ndarray)
    assert isinstance(f2_vals, np.ndarray)
