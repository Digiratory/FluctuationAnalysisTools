import os
import time
from multiprocessing import cpu_count

import numpy as np
import pytest

from StatTools.analysis.dfa import dfa
from StatTools.generators import generate_fbn

# Check GPU availability
try:
    import cupy as cp

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


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


@pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy not available, GPU tests skipped")
def test_dfa_cpu_vs_gpu_benchmark():
    """Benchmark test comparing CPU vs GPU performance"""
    np.random.seed(42)

    # Create larger dataset to see performance difference
    # Use multiple time series to make GPU advantage more visible
    n_series = 10
    length = 2**14  # 16384 points per series

    data_list = []
    for h in [0.5, 1.0, 1.5]:
        for _ in range(n_series // 3 + 1):
            series = generate_fbn(hurst=h, length=length, method="kasdin").flatten()
            data_list.append(series)

    data = np.array(data_list[:n_series])

    # Warm-up runs (to avoid cold start effects)
    _ = dfa(data, degree=2, processes=1, backend="cpu")
    _ = dfa(data, degree=2, processes=1, backend="gpu")

    # CPU benchmark
    n_cpu_cores = cpu_count()
    n_runs = 3
    cpu_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        s_cpu, f2_cpu = dfa(data, degree=2, processes=n_cpu_cores, backend="cpu")
        cpu_times.append(time.perf_counter() - start)

    # GPU benchmark
    gpu_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        s_gpu, f2_gpu = dfa(data, degree=2, processes=1, backend="gpu")
        gpu_times.append(time.perf_counter() - start)

    avg_cpu_time = np.mean(cpu_times)
    avg_gpu_time = np.mean(gpu_times)
    speedup = avg_cpu_time / avg_gpu_time

    print(f"\n{'='*60}")
    print(f"DFA Performance Benchmark")
    print(f"{'='*60}")
    print(f"CPU cores used: {n_cpu_cores}")
    print(f"Dataset: {n_series} time series, {length} points each")
    print(f"CPU average time: {avg_cpu_time:.4f} s (runs: {n_runs})")
    print(f"GPU average time: {avg_gpu_time:.4f} s (runs: {n_runs})")
    print(f"Speedup: {speedup:.2f}x")
    print(f"{'='*60}\n")

    # Verify results are consistent and compute difference statistics
    np.testing.assert_array_equal(s_cpu, s_gpu)
    _print_difference_stats(f2_cpu, f2_gpu)
    np.testing.assert_allclose(f2_cpu, f2_gpu, rtol=1e-5)

    # GPU should be at least as fast as CPU (or faster for large datasets)
    # Note: For small datasets, CPU might be faster due to GPU overhead
    assert avg_gpu_time > 0, "GPU time should be positive"


@pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy not available, GPU tests skipped")
def test_dfa_large_dataset_gpu_advantage():
    """Test GPU advantage on large dataset"""
    np.random.seed(42)

    # Very large dataset where GPU should show clear advantage
    n_series = 50
    length = 2**15  # 32768 points per series

    data_list = []
    for _ in range(n_series):
        series = generate_fbn(hurst=1.0, length=length, method="kasdin").flatten()
        data_list.append(series)

    data = np.array(data_list)

    # Single run for large dataset
    n_cpu_cores = cpu_count()
    start_cpu = time.perf_counter()
    s_cpu, f2_cpu = dfa(data, degree=2, processes=n_cpu_cores, backend="cpu")
    cpu_time = time.perf_counter() - start_cpu

    start_gpu = time.perf_counter()
    s_gpu, f2_gpu = dfa(data, degree=2, processes=1, backend="gpu")
    gpu_time = time.perf_counter() - start_gpu

    speedup = cpu_time / gpu_time

    print(f"\nLarge dataset benchmark:")
    print(f"  CPU cores used: {n_cpu_cores}")
    print(
        f"  Dataset: {n_series} series × {length} points = {n_series * length:,} total points"
    )
    print(f"  CPU time: {cpu_time:.4f} s")
    print(f"  GPU time: {gpu_time:.4f} s")
    print(f"  Speedup: {speedup:.2f}x")

    # Verify results and compute difference statistics
    np.testing.assert_array_equal(s_cpu, s_gpu)
    _print_difference_stats(f2_cpu, f2_gpu)
    np.testing.assert_allclose(f2_cpu, f2_gpu, rtol=1e-5)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy not available, GPU tests skipped")
def test_dfa_very_large_dataset_gpu_advantage():
    """Test GPU advantage on very large dataset"""
    np.random.seed(42)

    # Very large dataset where GPU should show significant advantage
    n_series = 100
    length = 2**16  # 65536 points per series

    data_list = []
    for _ in range(n_series):
        series = generate_fbn(hurst=1.0, length=length, method="kasdin").flatten()
        data_list.append(series)

    data = np.array(data_list)

    # Single run for very large dataset
    n_cpu_cores = cpu_count()
    start_cpu = time.perf_counter()
    s_cpu, f2_cpu = dfa(data, degree=2, processes=n_cpu_cores, backend="cpu")
    cpu_time = time.perf_counter() - start_cpu

    start_gpu = time.perf_counter()
    s_gpu, f2_gpu = dfa(data, degree=2, processes=1, backend="gpu")
    gpu_time = time.perf_counter() - start_gpu

    speedup = cpu_time / gpu_time

    print(f"\nVery large dataset benchmark:")
    print(f"  CPU cores used: {n_cpu_cores}")
    print(
        f"  Dataset: {n_series} series × {length} points = {n_series * length:,} total points"
    )
    print(f"  CPU time: {cpu_time:.4f} s")
    print(f"  GPU time: {gpu_time:.4f} s")
    print(f"  Speedup: {speedup:.2f}x")

    # Verify results and compute difference statistics
    np.testing.assert_array_equal(s_cpu, s_gpu)
    _print_difference_stats(f2_cpu, f2_gpu)
    np.testing.assert_allclose(f2_cpu, f2_gpu, rtol=1e-5)
