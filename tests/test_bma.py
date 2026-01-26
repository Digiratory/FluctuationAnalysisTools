import os

import numpy as np
import pytest
from scipy import stats

from StatTools.analysis.bma import bma
from StatTools.generators.kasdin_generator import create_kasdin_generator

# ------------------------------------------------------------
# PARAMETERS FROM TABLE
# ------------------------------------------------------------
IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

H_VALUES_CI = [0.3, 0.5, 0.8]
LENGTHS_CI = [2**14, 2**15]

H_VALUES = H_VALUES_CI if IN_GITHUB_ACTIONS else [0.3, 0.5, 0.8]
LENGTHS = LENGTHS_CI if IN_GITHUB_ACTIONS else [2**14, 2**15, 2**16]

# Fix seed to make test stable
np.random.seed(42)

# ------------------------------------------------------------
# fGn/fBm GENERATOR (Kasdin-style)
# ------------------------------------------------------------


def generate_fractional_noise(h: float, length: int):
    """Generate fGn using Kasdin filter method."""
    return create_kasdin_generator(h, length=length).get_full_sequence()


# ------------------------------------------------------------
# FIXTURE: multiple fGn/fBm signals for statistical tests
# ------------------------------------------------------------
@pytest.fixture(scope="module")
def test_dataset():
    """
    Produces dictionary:
        {
            ('fGn', H, N): [array10runs],
            ('fBm', H, N): [array10runs],
        }
    """
    data = {}
    for h in H_VALUES:
        for N in LENGTHS:
            fgn_runs = []
            fbm_runs = []
            for _ in range(3):
                fgn = generate_fractional_noise(h, N)
                fbm = np.cumsum(fgn)
                fgn_runs.append(fgn)
                fbm_runs.append(fbm)

            data[("fGn", h, N)] = fgn_runs
            data[("fBm", h, N)] = fbm_runs
    return data


# ------------------------------------------------------------
# UTILITY: estimate Hurst exponent
# ------------------------------------------------------------
def estimate_hurst(F, s):
    return stats.linregress(np.log(s), np.log(F)).slope


# ------------------------------------------------------------
# MAIN TEST: correctness of DMA/BMA estimator
# ------------------------------------------------------------
@pytest.mark.parametrize("h", H_VALUES)
@pytest.mark.parametrize("N", LENGTHS)
@pytest.mark.parametrize("rtype", ["fGn", "fBm"])
def test_bma_accuracy(test_dataset, h, N, rtype):
    """
    Compare estimated H vs true H.
    For fGn: n_integral = 1
    For fBm: n_integral = 0
    """
    runs = test_dataset[(rtype, h, N)]
    n_integral = 1 if rtype == "fGn" else 0

    scales = np.arange(10, N // 4, 5)

    H_estimates = []

    for signal in runs:
        F, s = bma(signal, s=scales, n_integral=n_integral, step=0.5)
        H_estimates.append(estimate_hurst(F, s))

    H_estimates = np.array(H_estimates)

    # mean (%)
    mean = np.abs(np.mean(np.abs(H_estimates)) - h)

    assert mean == pytest.approx(0, abs=0.2)
