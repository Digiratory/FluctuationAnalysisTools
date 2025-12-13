import os

import numpy as np
import pytest
from scipy import stats

from StatTools.analysis.svd_dfa import svd_dfa  # <-- твоя реализация SVD-DFA
from StatTools.generators.kasdin_generator import create_kasdin_generator

# ------------------------------------------------------------
# PARAMETERS (CI-friendly like in the example)
# ------------------------------------------------------------
IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

H_VALUES_CI = [0.3, 0.5, 0.8]
LENGTHS_CI = [2**10]
N_RUNS_CI = 10

H_VALUES = H_VALUES_CI if IN_GITHUB_ACTIONS else [0.3, 0.5, 0.8]
LENGTHS = LENGTHS_CI if IN_GITHUB_ACTIONS else [2**10, 2**14]

N_RUNS = N_RUNS_CI if IN_GITHUB_ACTIONS else [10]


# ------------------------------------------------------------
# GENERATOR (Kasdin-style)
# ------------------------------------------------------------
def generate_fractional_noise(h: float, length: int) -> np.ndarray:
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
    Where:
      - fGn is the raw generator output
      - fBm is cumulative sum of fGn
    """
    data = {}
    for h in H_VALUES:
        for N in LENGTHS:
            fgn_runs = []
            fbm_runs = []
            for _ in range(N_RUNS):
                fgn = generate_fractional_noise(h, N)
                fbm = np.cumsum(fgn)
                fgn_runs.append(fgn)
                fbm_runs.append(fbm)

            data[("fGn", h, N)] = fgn_runs
            data[("fBm", h, N)] = fbm_runs
    return data


# ------------------------------------------------------------
# UTILITY: estimate Hurst exponent (external, NOT inside method)
# ------------------------------------------------------------
def estimate_hurst(F: np.ndarray, s: np.ndarray) -> float:
    """Slope of log-log regression: log F(n) ~ H log n."""
    return stats.linregress(np.log(s), np.log(F)).slope


# ------------------------------------------------------------
# MAIN TEST: accuracy of SVD-DFA scaling exponent vs true H
# ------------------------------------------------------------
@pytest.mark.parametrize("h", H_VALUES)
@pytest.mark.parametrize("N", LENGTHS)
@pytest.mark.parametrize("rtype", ["fGn", "fBm"])
def test_svd_dfa_accuracy(test_dataset, h, N, rtype):
    """
    Compare estimated H vs true H.

    Important convention:
      - for fGn: integrate=True (DFA profile is required)
      - for fBm: integrate=False (already a profile)

    We check mean relative error (%) across multiple runs.
    Tolerance is deliberately wide (like in the sample tests),
    because DFA-like estimators are noisy on finite samples.
    """
    runs = test_dataset[(rtype, h, N)]
    integrate = True if rtype == "fGn" else False

    scales = np.arange(10, N // 4, 10)

    H_estimates = []
    for signal in runs:
        F, s = svd_dfa(
            signal,
            s=scales,
            integrate=integrate,
            L=None,  # default N//3
            p=2,  # typical
            m=1,  # DFA1 (linear detrending)
            n_min=10,
            n_max=N // 4,
        )
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


# ------------------------------------------------------------
# TEST: monotonicity — larger true H => larger estimated H (on average)
# ------------------------------------------------------------
@pytest.mark.parametrize("N", LENGTHS)
@pytest.mark.parametrize("rtype", ["fGn", "fBm"])
def test_svd_dfa_monotonic_in_h(test_dataset, N, rtype):
    """
    Sanity property:
      If H_true increases, the estimated scaling exponent should
      (on average) increase as well.

    This catches broken integration flags, swapped logs, etc.
    """
    integrate = True if rtype == "fGn" else False
    scales = np.arange(10, N // 4, 10)

    means = []
    for h in H_VALUES:
        runs = test_dataset[(rtype, h, N)]
        est = []
        for signal in runs:
            F, s = svd_dfa(signal, s=scales, integrate=integrate, p=2, m=1)
            est.append(estimate_hurst(F, s))
        means.append((h, float(np.mean(est))))

    # Check monotone non-decreasing
    for (h1, m1), (h2, m2) in zip(means, means[1:]):
        assert m2 >= m1 - 0.05, (
            f"Non-monotonic H estimates for {rtype}, N={N}: "
            f"H={h1}->{h2}, mean={m1:.3f}->{m2:.3f}"
        )


# ------------------------------------------------------------
# TEST: integration flag matters (fGn should be wrong without integration)
# ------------------------------------------------------------
@pytest.mark.parametrize("h", H_VALUES)
def test_svd_dfa_integration_flag_effect(h):
    """
    For fGn, integrate=True is the correct DFA convention.
    If we turn integration off, the estimate should typically shift.

    We only assert that the two estimates are meaningfully different,
    not which one is "closer", because finite-sample variability exists.
    """
    N = 2**14
    sig = generate_fractional_noise(h, N)
    scales = np.arange(10, N // 4, 10)

    F_on, s_on = svd_dfa(sig, s=scales, integrate=True, p=2, m=1)
    H_on = estimate_hurst(F_on, s_on)

    F_off, s_off = svd_dfa(sig, s=scales, integrate=False, p=2, m=1)
    H_off = estimate_hurst(F_off, s_off)

    diff = abs(H_on - H_off)
    assert diff >= 0.05, (
        f"Integration flag seems to have no effect for fGn (H={h}): "
        f"H_on={H_on:.3f}, H_off={H_off:.3f}, diff={diff:.3f}"
    )


# ------------------------------------------------------------
# TEST: p sensitivity should not explode (basic robustness)
# ------------------------------------------------------------
@pytest.mark.parametrize("h", H_VALUES)
def test_svd_dfa_p_sensitivity_reasonable(h):
    """
    p (number of removed SVD components) changes result,
    but estimates should stay within a reasonable band.

    This catches mistakes like removing all components or wrong reconstruction.
    """
    N = 2**14
    sig = generate_fractional_noise(h, N)
    scales = np.arange(10, N // 4, 10)

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