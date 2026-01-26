import multiprocessing
import os

import numpy as np
import pandas as pd
import pytest
from scipy import signal, stats

from StatTools.analysis.dpcca import dpcca
from StatTools.analysis.nonequidistant.dfa import dfa as ne_dfa

TEST_H_VALUES = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
TEST_H_VALUES_CI = [0.5, 1.0, 1.5]  # Reduced set for CI environments
IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

# Use reduced parameter set in GitHub Actions to speed up tests
h_target_values = TEST_H_VALUES_CI if IN_GITHUB_ACTIONS else TEST_H_VALUES


@pytest.fixture(scope="module")
def sample_signal():
    length = 2**13
    signals = {}
    # Generate signals only for the H values that will be tested
    test_h_values = h_target_values if IN_GITHUB_ACTIONS else TEST_H_VALUES
    for h in test_h_values:
        z = np.random.normal(size=length * 2)
        beta = 2 * h - 1
        L = length * 2
        A = np.zeros(length * 2)
        A[0] = 1
        for k in range(1, L):
            A[k] = (k - 1 - beta / 2) * A[k - 1] / k

        if h == 0.5:
            Z = z
        else:
            Z = signal.lfilter(1, A, z)
        signals[h] = Z

    return signals


@pytest.mark.timeout(300)  # 5 minutes timeout
@pytest.mark.parametrize("h_target", h_target_values)
def test_equidistant(sample_signal, h_target):
    """Test DFA with signals of known Hurst exponent"""
    np.random.seed(42)
    step = 0.5
    sig = sample_signal[h_target]
    s = [2**i for i in range(3, 13)]

    time_index = [pd.Timestamp(i, unit="s") for i in range(len(sig))]
    scales = [pd.Timedelta(i, unit="s") for i in s]

    ser = pd.Series(data=np.cumsum(sig), index=time_index)
    f1_ne_dfa = ne_dfa(ser, scales, step=step)
    f1_ne_dfa = np.sqrt(f1_ne_dfa.to_numpy())

    res_ne_dfa = stats.linregress(np.log(s), np.log(f1_ne_dfa))
    assert res_ne_dfa.slope == pytest.approx(h_target, 0.1)


@pytest.mark.timeout(300)  # 5 minutes timeout
@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test too long for Github Actions.")
@pytest.mark.parametrize("h_target", h_target_values)
def test_nonequidistant_0_5(sample_signal, h_target):
    """Test DFA with signals of known Hurst exponent"""
    np.random.seed(42)
    step = 0.5
    sig = sample_signal[h_target]
    s = [2**i for i in range(3, 13)]

    time_index = [pd.Timestamp(i, unit="s") for i in range(len(sig))]
    scales = [pd.Timedelta(i, unit="s") for i in s]

    ser = pd.Series(data=np.cumsum(sig), index=time_index)
    ser = ser.sample(frac=0.5, random_state=42).sort_index()

    f1_ne_dfa = ne_dfa(ser, scales, step=step)
    f1_ne_dfa = np.sqrt(f1_ne_dfa.to_numpy())

    res_ne_dfa = stats.linregress(np.log(s), np.log(f1_ne_dfa))
    assert res_ne_dfa.slope == pytest.approx(h_target, 0.1)


@pytest.mark.timeout(300)  # 5 minutes timeout
@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test too long for Github Actions.")
@pytest.mark.parametrize("h_target", h_target_values)
def test_nonequidistant_0_25_parallel_all_samples(sample_signal, h_target):
    """Test DFA with signals of known Hurst exponent"""
    np.random.seed(42)
    step = None
    sig = sample_signal[h_target]
    s = [2**i for i in range(3, 13)]

    time_index = [pd.Timestamp(i, unit="s") for i in range(len(sig))]
    scales = [pd.Timedelta(i, unit="s") for i in s]

    ser = pd.Series(data=np.cumsum(sig), index=time_index)
    ser = ser.sample(frac=0.25, random_state=42).sort_index()

    f1_ne_dfa = ne_dfa(ser, scales, step=step, num_process=multiprocessing.cpu_count())
    f1_ne_dfa = np.sqrt(f1_ne_dfa.to_numpy())

    res_ne_dfa = stats.linregress(np.log(s), np.log(f1_ne_dfa))
    assert res_ne_dfa.slope == pytest.approx(h_target, 0.1)
