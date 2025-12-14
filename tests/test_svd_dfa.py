import os

import numpy as np
import pytest
from scipy import stats

from StatTools.analysis.svd_dfa import svd_dfa
from StatTools.generators.kasdin_generator import create_kasdin_generator

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

H_VALUES_CI = [0.3, 0.5, 0.8]
N_CI = 2**13
N_RUNS_CI = 1
N_MAX_CI = 2**13

H_VALUES = H_VALUES_CI if IN_GITHUB_ACTIONS else [0.3, 0.5, 0.8]
N_LEN = N_CI if IN_GITHUB_ACTIONS else 2**13
N_RUNS = N_RUNS_CI if IN_GITHUB_ACTIONS else 1
N_MAX = N_MAX_CI if IN_GITHUB_ACTIONS else 2**13

"""
------------------------------------------------------------
GENERATOR (Kasdin-style)
------------------------------------------------------------
"""


def generate_fractional_noise(h: float, length: int) -> np.ndarray:
    """Generate fGn using Kasdin filter method."""
    return create_kasdin_generator(h, length=length).get_full_sequence()


"""
------------------------------------------------------------
FIXTURE: multiple fGn/fBm signals for statistical tests
------------------------------------------------------------
"""


@pytest.fixture(scope="module")
def test_dataset():
    """
    Produces dictionary:
        {
            ('fGn', H, N): [array10runs],
            ('fBm', H, N): [array10runs],
        }
    Where:
      - fGn is the raw generator output
      - fBm is cumulative sum of fGn
    """
    data = {}
    for h in H_VALUES:
        fgn_runs = []
        fbm_runs = []
        for _ in range(N_RUNS):
            fgn = generate_fractional_noise(h, N_LEN)
            fbm = np.cumsum(fgn)
            fgn_runs.append(fgn)
            fbm_runs.append(fbm)

        data[("fGn", h, N_LEN)] = fgn_runs
        data[("fBm", h, N_LEN)] = fbm_runs
    return data


"""
------------------------------------------------------------
UTILITY: estimate Hurst exponent (external, NOT inside method)
------------------------------------------------------------
"""


def estimate_hurst(F: np.ndarray, s: np.ndarray) -> float:
    """Slope of log-log regression: log F(n) ~ H log n."""
    return stats.linregress(np.log(s), np.log(F)).slope


"""
------------------------------------------------------------
MAIN TEST: accuracy of SVD-DFA scaling exponent vs true H
------------------------------------------------------------
"""


@pytest.mark.parametrize("h", H_VALUES)
@pytest.mark.parametrize("rtype", ["fGn", "fBm"])
def test_svd_dfa_accuracy(test_dataset, h, rtype):
    """
    Compare estimated H vs true H.

    Important convention:
      - for fGn: integrate=True (DFA profile is required)
      - for fBm: integrate=False (already a profile)

    We check mean relative error (%) across multiple runs.
    Tolerance is deliberately wide (like in the sample tests),
    because DFA-like estimators are noisy on finite samples.
    """
    N = N_LEN
    runs = test_dataset[(rtype, h, N)]
    integrate = True if rtype == "fGn" else False

    scales = np.array([2**i for i in range(3, 20)])

    H_estimates = []
    for signal in runs:
        F, s = svd_dfa(signal, s=scales, integrate=integrate, L=None, p=2, m=1)
        H_estimates.append(estimate_hurst(F, s))

    H_estimates = np.asarray(H_estimates)

    # Mean absolute relative error (%)
    rel_err = (H_estimates - h) / h
    eps = np.mean(np.abs(rel_err)) * 100

    tol_pct = 25
    assert eps == pytest.approx(0, abs=tol_pct), (
        f"SVD-DFA too inaccurate for {rtype}, H={h}, N={N}: "
        f"mean abs rel err = {eps:.1f}% >= {tol_pct}%"
    )


"""
------------------------------------------------------------
 TEST: p sensitivity should not explode (basic robustness)
------------------------------------------------------------
"""


@pytest.mark.parametrize("h", H_VALUES)
def test_svd_dfa_p_sensitivity_reasonable(h):
    """
    p (number of removed SVD components) changes result,
    but estimates should stay within a reasonable band.

    This catches mistakes like removing all components or wrong reconstruction.
    """
    N = N_MAX
    sig = generate_fractional_noise(h, N)
    scales = np.array([2**i for i in range(3, 20)])

    # correct convention for fGn
    estimates = []
    for p in [0, 1, 2, 3]:
        F, s = svd_dfa(sig, s=scales, integrate=True, p=p, m=1)
        estimates.append(estimate_hurst(F, s))

    estimates = np.asarray(estimates)
    spread = float(np.max(estimates) - np.min(estimates))

    assert spread <= 0.35, (
        f"Too sensitive to p for H={h}: "
        f"estimates={estimates}, spread={spread:.3f} > 0.35"
    )
