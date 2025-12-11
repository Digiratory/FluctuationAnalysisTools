import os

import numpy as np
import pytest
from scipy import stats

from StatTools.analysis import dpcca
from StatTools.analysis.bma import bma
from StatTools.generators.kasdin_generator import create_kasdin_generator

# ------------------------------------------------------------
# PARAMETERS FROM TABLE
# ------------------------------------------------------------
IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

H_VALUES_CI = [0.3, 0.5, 0.8]
LENGTHS_CI = [2**10, 2**14]

H_VALUES = H_VALUES_CI if IN_GITHUB_ACTIONS else [0.3, 0.5, 0.8]
LENGTHS = LENGTHS_CI if IN_GITHUB_ACTIONS else [2**10, 2**14]


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
            for _ in range(10):
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

    # Relative error mean (%)
    rel_err = (H_estimates - h) / h
    eps = np.mean(np.abs(rel_err)) * 100

    tol = 25

    assert eps == pytest.approx(0, abs=tol)


@pytest.mark.parametrize("h", H_VALUES)
def test_bma_vs_dfa(h):
    """
    Compare BMA DPCCA implementation on fractional Gaussian noise (fGn).

    Steps:
    - Generate fGn signal with known Hurst exponent `h`
    - Compute Hurst exponent using our BMA-based DFA method (`H_bma`)
    - Compute Hurst exponent using `DPCCA` (`H_dpcca`)
    - Assert that both estimates are close within a reasonable tolerance

    """
    N = 2**14  # Use sufficiently long signal for stable DFA estimation

    # Generate fractional Gaussian noise with target Hurst exponent
    sig = generate_fractional_noise(h, N)

    # Compute DFA using our BMA method
    scales = np.arange(
        10, N // 4, 10
    )  # Define scale range, avoiding too small/large boxes
    F_bma, s_bma = bma(sig, s=scales, n_integral=1, step=0.5)  # Perform BMA DFA
    H_bma = estimate_hurst(F_bma, s_bma)  # Estimate Hurst from scaling exponent

    # Compute DPCCA using dpcca method
    P, R, F, S = dpcca(sig, 2, 0.5, scales, len(scales), n_integral=1)
    F = np.sqrt(F)
    H_dpcca = estimate_hurst(F, S)

    diff = abs(H_bma - H_dpcca)
    diff_pct = diff / H_dpcca * 100

    # Assert that both methods yield similar results (tolerance accounts for methodological differences)
    assert diff_pct < 25, (
        f"BMA and dfa() differ too much for H={h}: "
        f"|{H_bma:.3f} - {H_dpcca:.3f}| = {diff_pct:.1f}% >= 25%"
    )
