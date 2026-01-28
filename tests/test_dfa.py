import os
import warnings

import numpy as np
import pytest
from scipy import signal, stats

from StatTools.analysis.dfa import DFA, _detrend_segment, dfa, dfa_worker
from StatTools.generators import generate_fbn

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

# ---------------------- H values (shared) ----------------------
TEST_H_VALUES_FULL = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
TEST_H_VALUES_CI = [0.5, 1.0, 1.5]

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


@pytest.fixture(scope="module")
def sample_signals():
    """Generate sample signals with different Hurst exponents for testing"""
    length = 2**13
    signals = {}
    np.random.seed(42)  # Set seed for reproducibility
    for h in TEST_H_VALUES:
        signals[h] = generate_fbn(hurst=h, length=length, method="kasdin").flatten()
    return signals


def test_detrend_segment():
    """Test _detrend_segment function"""
    indices = np.arange(10)
    y_segment = 2 * indices + 1 + np.random.normal(0, 0.1, 10)

    residuals = _detrend_segment(y_segment, indices, degree=1)

    assert isinstance(residuals, np.ndarray)
    assert residuals.shape == (10,)
    assert np.abs(np.mean(residuals)) < 0.5


def test_detrend_segment_quadratic():
    """Test _detrend_segment with quadratic trend"""
    indices = np.arange(20)
    y_segment = indices**2 + 3 * indices + 1 + np.random.normal(0, 0.1, 20)

    residuals = _detrend_segment(y_segment, indices, degree=2)

    assert isinstance(residuals, np.ndarray)
    assert residuals.shape == (20,)
    assert np.abs(np.mean(residuals)) < 1.0


def test_dfa_worker_1d_input():
    """Test dfa_worker with single series."""
    np.random.seed(42)
    data_2d = generate_fbn(hurst=0.5, length=1000, method="kasdin")

    result_cpu = dfa_worker(indices=0, arr=data_2d, degree=2, backend="cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result_gpu = dfa_worker(indices=0, arr=data_2d, degree=2, backend="gpu")

    assert isinstance(result_cpu, list)
    assert len(result_cpu) == 1
    assert isinstance(result_gpu, list)
    assert len(result_gpu) == 1

    s_cpu, f2_cpu = result_cpu[0]
    s_gpu, f2_gpu = result_gpu[0]

    assert isinstance(s_cpu, np.ndarray)
    assert isinstance(f2_cpu, np.ndarray)
    assert len(s_cpu) == len(f2_cpu)
    assert len(s_cpu) > 0

    np.testing.assert_array_equal(s_cpu, s_gpu)
    np.testing.assert_allclose(f2_cpu, f2_gpu, rtol=1e-5, atol=1e-12)


def test_dfa_worker_2d_input():
    """Test dfa_worker with multiple series."""
    np.random.seed(42)
    data_list = []
    for h in [0.5, 1.0, 1.5]:
        series = generate_fbn(hurst=h, length=1000, method="kasdin").flatten()
        data_list.append(series)
    data = np.array(data_list)

    result_cpu = dfa_worker(indices=[0, 1, 2], arr=data, degree=2, backend="cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result_gpu = dfa_worker(indices=[0, 1, 2], arr=data, degree=2, backend="gpu")

    assert isinstance(result_cpu, list)
    assert len(result_cpu) == 3
    assert isinstance(result_gpu, list)
    assert len(result_gpu) == 3

    for (s_cpu, f2_cpu), (s_gpu, f2_gpu) in zip(result_cpu, result_gpu):
        assert isinstance(s_cpu, np.ndarray)
        assert isinstance(f2_cpu, np.ndarray)
        assert len(s_cpu) == len(f2_cpu)
        np.testing.assert_array_equal(s_cpu, s_gpu)
        np.testing.assert_allclose(f2_cpu, f2_gpu, rtol=1e-5, atol=1e-12)


def test_dfa_worker_custom_s_values():
    """Test dfa_worker with custom s_values."""
    np.random.seed(42)
    data_2d = generate_fbn(hurst=0.5, length=1000, method="kasdin")
    custom_s = [16, 32, 64, 128]

    result_cpu = dfa_worker(
        indices=0, arr=data_2d, degree=2, s_values=custom_s, backend="cpu"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result_gpu = dfa_worker(
            indices=0, arr=data_2d, degree=2, s_values=custom_s, backend="gpu"
        )

    s_cpu, f2_cpu = result_cpu[0]
    s_gpu, f2_gpu = result_gpu[0]

    assert len(s_cpu) <= len(custom_s)
    assert all(int(s) in custom_s for s in s_cpu)

    np.testing.assert_array_equal(s_cpu, s_gpu)
    np.testing.assert_allclose(f2_cpu, f2_gpu, rtol=1e-5, atol=1e-12)


def test_dfa_worker_invalid_dimension():
    """Test dfa_worker raises error for invalid dimensions"""
    data_3d = np.random.normal(0, 1, (5, 5, 5))

    with pytest.raises(ValueError, match="expects 2D array"):
        dfa_worker(indices=0, arr=data_3d, degree=2)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("h", TEST_H_VALUES)
def test_dfa_1d_with_known_h(sample_signals, h, backend):
    """Test dfa function with 1D input and known Hurst exponent"""
    sig = sample_signals[h]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        s_vals, f2_vals = dfa(sig, degree=2, processes=1, backend=backend)

    assert isinstance(s_vals, np.ndarray)
    assert isinstance(f2_vals, np.ndarray)
    assert len(s_vals) == len(f2_vals)
    assert len(s_vals) > 0

    if h > 0.5:
        assert np.all(np.diff(f2_vals) > 0) or np.all(np.diff(f2_vals) >= -1e-10)

    log_s = np.log(s_vals)
    log_f = np.log(np.sqrt(f2_vals))
    res = stats.linregress(log_s, log_f)

    assert res.slope == pytest.approx(h, rel=0.15)


@pytest.mark.parametrize("h", TEST_H_VALUES)
def test_dfa_1d_parallel(sample_signals, h):
    """Test dfa function with parallel processing"""
    sig = sample_signals[h]

    s_vals, f2_vals = dfa(sig, degree=2, processes=2)

    assert isinstance(s_vals, np.ndarray)
    assert isinstance(f2_vals, np.ndarray)
    assert len(s_vals) == len(f2_vals)

    log_s = np.log(s_vals)
    log_f = np.log(np.sqrt(f2_vals))
    res = stats.linregress(log_s, log_f)

    assert res.slope == pytest.approx(h, rel=0.15)


def test_dfa_2d_input():
    """Test dfa function with 2D input (multiple time series)"""
    np.random.seed(42)
    data_list = []
    h_list = []
    for h in TEST_H_VALUES:
        series = generate_fbn(hurst=h, length=1000, method="kasdin").flatten()
        data_list.append(series)
        h_list.append(h)
    data = np.array(data_list)

    s_vals_cpu, f2_vals_cpu = dfa(data, degree=2, processes=1, backend="cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        s_vals_gpu, f2_vals_gpu = dfa(data, degree=2, processes=1, backend="gpu")

    assert isinstance(s_vals_cpu, np.ndarray)
    assert isinstance(f2_vals_cpu, np.ndarray)
    assert f2_vals_cpu.ndim == 2
    assert f2_vals_cpu.shape[0] == len(TEST_H_VALUES)
    assert f2_vals_cpu.shape[1] == len(s_vals_cpu)

    np.testing.assert_array_equal(s_vals_cpu, s_vals_gpu)
    np.testing.assert_allclose(f2_vals_cpu, f2_vals_gpu, rtol=1e-5, atol=1e-12)

    log_s = np.log(s_vals_cpu)
    for i, h_target in enumerate(h_list):
        log_f = np.log(np.sqrt(f2_vals_cpu[i]))
        res = stats.linregress(log_s, log_f)
        assert res.slope == pytest.approx(h_target, rel=0.15)


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

    log_s = np.log(s_vals)
    for i, h_target in enumerate(h_list):
        log_f = np.log(np.sqrt(f2_vals[i]))
        res = stats.linregress(log_s, log_f)
        assert res.slope == pytest.approx(h_target, rel=0.15)


def test_dfa_different_degrees():
    """Test dfa function with different polynomial degrees"""
    np.random.seed(42)
    data = generate_fbn(hurst=1.0, length=1000, method="kasdin").flatten()

    s1, f2_1 = dfa(data, degree=1, processes=1)
    s2, f2_2 = dfa(data, degree=2, processes=1)
    s3, f2_3 = dfa(data, degree=3, processes=1)

    assert len(s1) > 0
    assert len(s2) > 0
    assert len(s3) > 0

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
    data = generate_fbn(hurst=1.0, length=10, method="kasdin").flatten()

    s_vals, f2_vals = dfa(data, degree=2)

    assert isinstance(s_vals, np.ndarray)
    assert isinstance(f2_vals, np.ndarray)


def test_gpu_backend_request_does_not_crash_and_matches_cpu():
    """Requesting backend='gpu' should produce a valid result."""
    np.random.seed(42)
    data = generate_fbn(hurst=1.0, length=2000, method="kasdin").flatten()

    s_cpu, f2_cpu = dfa(data, degree=2, processes=1, backend="cpu")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        s_gpu_req, f2_gpu_req = dfa(data, degree=2, processes=1, backend="gpu")

    np.testing.assert_array_equal(s_cpu, s_gpu_req)
    np.testing.assert_allclose(f2_cpu, f2_gpu_req, rtol=1e-5, atol=1e-12)
