import os

import nolds
import numpy as np
import pytest
from scipy import signal, stats

from StatTools.analysis.bma import bma

# ------------------------------------------------------------
# PARAMETERS FROM TABLE
# ------------------------------------------------------------
IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

H_VALUES_CI = [0.2, 0.8]
LENGTHS_CI = [2**10, 2**16]

H_VALUES = H_VALUES_CI if IN_GITHUB_ACTIONS else [0.2, 0.4, 0.5, 0.6, 0.8]
LENGTHS = LENGTHS_CI if IN_GITHUB_ACTIONS else [2**10, 2**12, 2**14, 2**16]


# ------------------------------------------------------------
# fGn/fBm GENERATOR (Kasdin-style)
# ------------------------------------------------------------


def generate_fractional_noise(h: float, length: int):
    """Generate fGn using Kasdin filter method."""
    z = np.random.normal(size=length * 2)
    beta = 2 * h - 1
    L = length * 2
    A = np.zeros(L)
    A[0] = 1
    for k in range(1, L):
        A[k] = (k - 1 - beta / 2) * A[k - 1] / k

    if h == 0.5:  # white noise case
        Z = z
    else:
        Z = signal.lfilter(1, A, z)

    return Z[:length]


def generate_fbm(h: float, length: int):
    """Integrate fGn to produce fBm."""
    return np.cumsum(generate_fractional_noise(h, length))


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
def test_dma_accuracy(test_dataset, h, N, rtype):
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

    # Absolute error mean
    eps = np.mean(np.abs(H_estimates - h))

    # RMSE
    rmse = np.sqrt(np.mean((H_estimates - h) ** 2))

    print(f"\nType={rtype} H={h} N={N} eps={eps:.3f} rmse={rmse:.3f}")

    # Assertions:
    #  - BMA is less accurate than CDMA, so tolerances are wider than for FA
    #  - For large N error must be small
    tol = 0.22 if N < 4000 else 0.15

    assert eps == pytest.approx(0, abs=tol)
    assert rmse == pytest.approx(0, abs=tol)


@pytest.mark.parametrize("h", H_VALUES)
def test_bma_vs_nolds_dfa(h):
    """
    Сравнение с nolds.dfa():

    - генерируем fGn с заданным H
    - считаем H_bma нашим методом
    - считаем H_nolds через nolds.dfa
    - проверяем, что они достаточно близки
    """
    N = 2**14

    sig = generate_fractional_noise(h, N)

    # Наш BMA
    scales = np.arange(10, N // 4, 10)
    F_bma, s_bma = bma(sig, s=scales, n_integral=1, step=0.5)
    H_bma = estimate_hurst(F_bma, s_bma)

    # nolds.dfa
    # dfa возвращает показатель альфа, который для fGn интерпретируется как H
    H_nolds = nolds.dfa(sig)

    print(
        f"\nH_true={h:.2f}  H_bma={H_bma:.3f}  "
        f"H_nolds={H_nolds:.3f}  diff={abs(H_bma - H_nolds):.3f}"
    )

    # допускаем некоторое различие, но методы не должны сильно расходиться
    assert abs(H_bma - H_nolds) < 0.2
