import os

import numpy as np
import pytest
from scipy import stats

from StatTools.analysis.dfa import dfa, dfa_worker
from StatTools.generators import generate_fbn

# Check GPU availability
try:
    import cupy as cp

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
if IN_GITHUB_ACTIONS:
    TEST_H_VALUES = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
else:
    TEST_H_VALUES = [0.5, 1.0, 1.5, 2.0]


def _print_difference_stats(f2_cpu, f2_gpu, label="CPU vs GPU"):
    """Compute and print difference statistics between CPU and GPU results"""
    abs_diff = np.abs(f2_cpu - f2_gpu)
    rel_diff = abs_diff / (
        np.abs(f2_cpu) + 1e-15
    )  # Add epsilon to avoid division by zero

    mean_abs_diff = np.mean(abs_diff)
    max_abs_diff = np.max(abs_diff)
    mean_rel_diff = np.mean(rel_diff)
    max_rel_diff = np.max(rel_diff)

    print(f"{label} difference statistics:")
    print(f"  Mean absolute difference: {mean_abs_diff:.2e}")
    print(f"  Max absolute difference: {max_abs_diff:.2e}")
    print(f"  Mean relative difference: {mean_rel_diff:.2e}")
    print(f"  Max relative difference: {max_rel_diff:.2e}")


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


# ====================== Tests for dfa_worker ======================


@pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy not available, GPU tests skipped")
def test_dfa_worker_1d_input_gpu():
    """Test dfa_worker with 1D input and GPU acceleration"""
    np.random.seed(42)
    # generate_fbn returns (1, length), we need 2D for dfa_worker
    data_2d = generate_fbn(hurst=0.5, length=1000, method="kasdin")

    result = dfa_worker(indices=0, arr=data_2d, degree=2, backend="gpu")

    assert isinstance(result, list)
    assert len(result) == 1
    s_vals, f2_vals = result[0]
    assert isinstance(s_vals, np.ndarray)
    assert isinstance(f2_vals, np.ndarray)
    assert len(s_vals) == len(f2_vals)
    assert len(s_vals) > 0


@pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy not available, GPU tests skipped")
def test_dfa_worker_2d_input_gpu():
    """Test dfa_worker with 2D input and GPU acceleration"""
    np.random.seed(42)
    # Generate multiple time series
    data_list = []
    for h in [0.5, 1.0, 1.5]:
        series = generate_fbn(hurst=h, length=1000, method="kasdin").flatten()
        data_list.append(series)
    data = np.array(data_list)

    result = dfa_worker(indices=[0, 1, 2], arr=data, degree=2, backend="gpu")

    assert isinstance(result, list)
    assert len(result) == 3
    for s_vals, f2_vals in result:
        assert isinstance(s_vals, np.ndarray)
        assert isinstance(f2_vals, np.ndarray)
        assert len(s_vals) == len(f2_vals)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy not available, GPU tests skipped")
def test_dfa_worker_custom_s_values_gpu():
    """Test dfa_worker with custom s_values and GPU acceleration"""
    np.random.seed(42)
    data_2d = generate_fbn(hurst=0.5, length=1000, method="kasdin")
    custom_s = [16, 32, 64, 128]

    result = dfa_worker(
        indices=0, arr=data_2d, degree=2, s_values=custom_s, backend="gpu"
    )

    s_vals, f2_vals = result[0]
    assert len(s_vals) <= len(custom_s)  # Some may be filtered out
    assert all(s in custom_s for s in s_vals)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy not available, GPU tests skipped")
def test_dfa_worker_invalid_dimension_gpu():
    """Test dfa_worker raises error for invalid dimensions with GPU"""
    data_3d = np.random.normal(0, 1, (5, 5, 5))

    with pytest.raises(ValueError, match="expects 2D array"):
        dfa_worker(indices=0, arr=data_3d, degree=2, backend="gpu")


# ====================== Tests for dfa function ======================


@pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy not available, GPU tests skipped")
@pytest.mark.parametrize("h", TEST_H_VALUES)
def test_dfa_1d_with_known_h_gpu(sample_signals, h):
    """Test dfa function with 1D input, known Hurst exponent, and GPU acceleration"""
    sig = sample_signals[h]

    s_vals, f2_vals = dfa(sig, degree=2, processes=1, backend="gpu")

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

    assert res.slope == pytest.approx(h, rel=0.15)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy not available, GPU tests skipped")
def test_dfa_1d_gpu_warns_on_multiprocessing():
    """Test that dfa warns when backend='gpu' and processes > 1"""
    np.random.seed(42)
    sig = generate_fbn(hurst=0.5, length=1000, method="kasdin").flatten()

    with pytest.warns(UserWarning, match="multiprocessing.*disabled"):
        s_vals, f2_vals = dfa(sig, degree=2, processes=2, backend="gpu")

    assert isinstance(s_vals, np.ndarray)
    assert isinstance(f2_vals, np.ndarray)
    assert len(s_vals) == len(f2_vals)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy not available, GPU tests skipped")
def test_dfa_2d_input_gpu():
    """Test dfa function with 2D input (multiple time series) and GPU acceleration"""
    np.random.seed(42)
    # Generate multiple time series with different H
    data_list = []
    h_list = []
    for h in TEST_H_VALUES:
        series = generate_fbn(hurst=h, length=1000, method="kasdin").flatten()
        data_list.append(series)
        h_list.append(h)
    data = np.array(data_list)

    s_vals, f2_vals = dfa(data, degree=2, processes=1, backend="gpu")

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
        # Use relative tolerance like in test_utils.py (15% relative tolerance)
        assert res.slope == pytest.approx(h_target, rel=0.15)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy not available, GPU tests skipped")
def test_dfa_different_degrees_gpu():
    """Test dfa function with different polynomial degrees and GPU acceleration"""
    np.random.seed(42)
    data = generate_fbn(hurst=1.0, length=1000, method="kasdin").flatten()

    s1, f2_1 = dfa(data, degree=1, processes=1, backend="gpu")
    s2, f2_2 = dfa(data, degree=2, processes=1, backend="gpu")
    s3, f2_3 = dfa(data, degree=3, processes=1, backend="gpu")

    # All should return valid results
    assert len(s1) > 0
    assert len(s2) > 0
    assert len(s3) > 0

    # Results should be similar but not identical
    assert len(s1) == len(s2) == len(s3)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy not available, GPU tests skipped")
def test_dfa_empty_input_gpu():
    """Test dfa function with empty input and GPU acceleration"""
    with pytest.raises(ValueError):
        dfa(np.array([]), degree=2, backend="gpu")


@pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy not available, GPU tests skipped")
def test_dfa_invalid_dimension_gpu():
    """Test dfa function with invalid dimensions and GPU acceleration"""
    data_3d = np.random.normal(0, 1, (5, 5, 5))

    with pytest.raises(ValueError, match="Only 1D or 2D arrays"):
        dfa(data_3d, degree=2, backend="gpu")


@pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy not available, GPU tests skipped")
def test_dfa_short_input_gpu():
    """Test dfa function with too short input and GPU acceleration"""
    np.random.seed(42)
    data = generate_fbn(hurst=1.0, length=10, method="kasdin").flatten()  # Very short

    # Should not raise error, but may return empty or very few scales
    s_vals, f2_vals = dfa(data, degree=2, backend="gpu")

    # Should still return valid arrays (may be empty or very short)
    assert isinstance(s_vals, np.ndarray)
    assert isinstance(f2_vals, np.ndarray)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy not available, GPU tests skipped")
def test_dfa_gpu_vs_cpu_consistency():
    """Test that GPU and CPU produce consistent results"""
    np.random.seed(42)
    data = generate_fbn(hurst=1.0, length=2000, method="kasdin").flatten()

    # Run on CPU
    s_cpu, f2_cpu = dfa(data, degree=2, processes=1, backend="cpu")

    # Run on GPU
    s_gpu, f2_gpu = dfa(data, degree=2, processes=1, backend="gpu")

    # Results should be very similar (allowing for small numerical differences)
    np.testing.assert_array_equal(s_cpu, s_gpu)
    _print_difference_stats(f2_cpu, f2_gpu)
    np.testing.assert_allclose(f2_cpu, f2_gpu, rtol=1e-5)
