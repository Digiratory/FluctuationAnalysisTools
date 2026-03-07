"""
EMD-based Multifractal Detrended Fluctuation Analysis (EMD-MFDFA)

Implements the EMD-based (MF)DFA algorithm of Qian, Gu & Zhou (2011).
The only modification to the classical MFDFA is Step 3: polynomial
detrending is replaced by EMD-based detrending applied to each segment
independently (Section 2.4, Eq. 13).

Reference:
    Xi-Yuan Qian, Gao-Feng Gu, Wei-Xing Zhou (2011). Modified detrended
    fluctuation analysis based on empirical mode decomposition for the
    characterization of anti-persistent processes. Physica A, 390(23-24),
    4388-4395.  DOI: https://doi.org/10.1016/j.physa.2011.07.008
"""

import warnings
from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import CubicSpline


def _find_extrema(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Find indices of local maxima and minima via first differences."""
    d = np.diff(signal)
    maxima = np.where((d[:-1] > 0) & (d[1:] <= 0))[0] + 1
    minima = np.where((d[:-1] < 0) & (d[1:] >= 0))[0] + 1
    return maxima, minima


def _make_envelope(x_points: np.ndarray, y_points: np.ndarray, n: int) -> np.ndarray:
    """
    Build an envelope through extrema via cubic spline with mirror
    boundary extension (standard EMD practice, cf. Rilling et al. 2003).
    """
    if len(x_points) <= 1:
        return np.full(n, np.mean(y_points) if len(y_points) > 0 else 0.0)

    xl = list(x_points.astype(float))
    yl = list(y_points.astype(float))

    if x_points[0] > 0:
        xl.insert(0, float(-x_points[0]))
        yl.insert(0, float(y_points[0]))

    if x_points[-1] < n - 1:
        xl.append(float(2 * (n - 1) - x_points[-1]))
        yl.append(float(y_points[-1]))

    x_ext = np.asarray(xl)
    y_ext = np.asarray(yl)

    _, ui = np.unique(x_ext, return_index=True)
    x_ext, y_ext = x_ext[ui], y_ext[ui]

    return CubicSpline(x_ext, y_ext, bc_type="natural")(np.arange(n))


def _emd_decompose(
    signal: np.ndarray,
    max_imfs: int = 10,
    max_siftings: int = 100,
    sd_threshold: float = 0.2,
) -> np.ndarray:
    """
    Empirical Mode Decomposition (Section 2.3, steps 1-6, Eq. 10-12).

    Returns array whose rows are [IMF_1, ..., IMF_n, residual].
    The last row is always the residual r_n representing the local trend.
    """
    n = len(signal)
    residual = signal.astype(float, copy=True)
    imfs = []

    for _ in range(max_imfs):
        h = residual.copy()

        sifted = False
        for _ in range(max_siftings):
            maxima, minima = _find_extrema(h)

            if len(maxima) < 2 or len(minima) < 2:
                break

            upper = _make_envelope(maxima, h[maxima], n)
            lower = _make_envelope(minima, h[minima], n)
            mean_env = (upper + lower) / 2.0  # Eq. 10

            prev_h = h.copy()
            h = h - mean_env  # Eq. 11
            sifted = True

            sd = np.sum((prev_h - h) ** 2) / (np.sum(prev_h**2) + 1e-12)
            if sd < sd_threshold:
                break

        if not sifted:
            break

        imfs.append(h)
        residual = residual - h  # toward Eq. 12

        maxima, minima = _find_extrema(residual)
        if len(maxima) + len(minima) < 2:
            break

    imfs.append(residual)  # r_n (Eq. 12)
    return np.array(imfs)


def _emd_detrend_segment(segment: np.ndarray) -> np.ndarray:
    """
    EMD-based detrending of a single segment (Eq. 13).

    The trend r_n is determined for each segment separately.
    """
    imfs = _emd_decompose(segment)
    trend = imfs[-1]
    return segment - trend


def _emd_mfdfa_fluctuations(
    signal: np.ndarray,
    q_values: np.ndarray,
    scales: np.ndarray,
    n_integral: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Steps 1-5 of the EMD-based MFDFA (Sections 2.1-2.2 + 2.4).

    Returns (h_q, Fq).
    """
    N = len(signal)

    # Step 1: cumulative sum — Eq. 1
    profile = signal - np.mean(signal)
    for _ in range(n_integral):
        profile = np.cumsum(profile)

    Fq = np.zeros((len(q_values), len(scales)))

    for s_idx, s in enumerate(scales):
        # Step 2: partition into N_s segments of size s
        n_segments = N // s
        if n_segments < 1:
            continue

        # Step 3 + Eq. 3: EMD detrending → segment variance
        F2 = np.empty(n_segments)
        for v in range(n_segments):
            segment = profile[v * s : (v + 1) * s]
            residuals = _emd_detrend_segment(segment)
            F2[v] = np.mean(residuals**2)

        # Step 4 — Eq. 6, 7
        F_vs = np.sqrt(np.maximum(F2, 1e-20))
        for q_idx, q in enumerate(q_values):
            if q == 0:
                Fq[q_idx, s_idx] = np.exp(np.mean(np.log(F_vs)))
            else:
                Fq[q_idx, s_idx] = np.power(np.mean(np.power(F_vs, q)), 1.0 / q)

    # Step 5 — Eq. 8: h(q) from log-log slope
    h_q = np.zeros(len(q_values))
    for q_idx in range(len(q_values)):
        valid = Fq[q_idx] > 0
        if np.sum(valid) > 2:
            coeffs = np.polyfit(np.log(scales[valid]), np.log(Fq[q_idx, valid]), 1)
            h_q[q_idx] = coeffs[0]

    return h_q, Fq


def emd_mfdfa(
    signal: np.ndarray,
    q_values: Optional[np.ndarray] = None,
    scales: Optional[np.ndarray] = None,
    n_integral: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    EMD-based Multifractal Detrended Fluctuation Analysis.

    Implements the algorithm of Qian, Gu & Zhou (2011): the classical
    MFDFA procedure where the polynomial detrending step (Step 3) is
    replaced by EMD-based detrending applied to each segment independently.

    Basic usage::

        import numpy as np
        from StatTools.analysis.emd_mfdfa import emd_mfdfa
        from StatTools.generators.kasdin_generator import create_kasdin_generator

        generator = create_kasdin_generator(h=0.7, length=2**12)
        signal = generator.get_full_sequence()

        q, h_q, scales = emd_mfdfa(signal)
        h_estimate = h_q[np.where(q == 2)[0][0]]
        print(f"Estimated Hurst exponent: {h_estimate:.3f}")

    Args:
        signal (np.ndarray): Input time series (1D array).
        q_values (np.ndarray, optional): Array of q-orders for multifractal
            analysis.  Default: ``np.arange(-5, 6, 1)``.
        scales (np.ndarray, optional): Array of time scales (segment sizes).
            Default: logarithmically spaced from 10 to N/4.
        n_integral (int): Number of cumulative-sum (integration) operations
            applied before the analysis (default: 1).

    Returns:
        tuple: (q_values, h_q, scales) where

            - *q_values* — q-order values used,
            - *h_q* — generalized Hurst exponents h(q),
            - *scales* — time scales used.

    Raises:
        ValueError: If the signal is too short or has wrong dimensions.

    Notes:
        - For fGn (fractional Gaussian noise): h(q) ≈ const (monofractal).
        - For fBm (fractional Brownian motion): use ``n_integral=0``.
        - h(q=2) corresponds to the standard Hurst exponent H.

    See Also:
        StatTools.analysis.dfa : Classical polynomial-based DFA.
    """
    if not isinstance(signal, np.ndarray):
        signal = np.array(signal)

    if signal.ndim != 1:
        raise ValueError("Input signal must be a 1D array")

    N = len(signal)
    if N < 64:
        raise ValueError(
            f"Signal too short (N={N}). "
            f"With s_max=N/4={N // 4} the log-log fitting range is too narrow."
        )

    if N < 1024:
        warnings.warn(
            f"Signal length (N={N}) may be too short for reliable scaling "
            f"estimation: with s_max=N/4={N // 4}, the log-log fitting range "
            "is narrow.",
            UserWarning,
            stacklevel=2,
        )

    if q_values is None:
        q_values = np.arange(-5, 6, 1)

    if scales is None:
        s_min = 16
        s_max = N // 4
        scales = np.unique(
            np.logspace(np.log10(s_min), np.log10(s_max), num=25, dtype=int)
        )

    h_q, _ = _emd_mfdfa_fluctuations(signal, q_values, scales, n_integral)

    return q_values, h_q, scales
