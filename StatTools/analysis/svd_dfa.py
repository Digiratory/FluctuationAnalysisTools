from __future__ import annotations

from collections.abc import Iterable
from typing import Optional, Tuple, Union

import numpy as np

# ---------------------------
# Helpers: profile, SVD filter
# ---------------------------


def _as_2d(arr: np.ndarray) -> np.ndarray:
    """Convert input to shape (n_signals, N)."""
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        return arr[None, :]
    if arr.ndim == 2:
        return arr
    raise ValueError("arr must be 1D or 2D array")


def _integral_profile(x: np.ndarray) -> np.ndarray:
    """
    Integral profile for DFA:
        y(t) = sum_{i<=t} (x(i) - mean(x))

    Notes:
      - The mean is removed per signal (row-wise), so the profile corresponds to the
        cumulative sum of zero-mean fluctuations.
      - Vectorized over signals: input and output have shape (n_signals, N).
      - This "integration" step is standard in DFA and helps convert the original
        series into a random-walk-like profile where scaling of detrended fluctuations
        is more stable and interpretable.
    """
    x = x - np.mean(x, axis=1, keepdims=True)
    return np.cumsum(x, axis=1)


def _trajectory_matrix(y: np.ndarray, L: int) -> np.ndarray:
    """
    Build the trajectory (Hankel) matrix from a profile y:

        Y[i, j] = y[i + j],    i = 0..L-1, j = 0..K-1
        K = N - L + 1

    Intuition:
      - This is the standard "embedding" step used in SSA/SVD-based filtering.
      - Each column of Y is a length-L window of the profile; consecutive columns
        are shifted by one sample.

    Input:
      y: shape (n_signals, N)
      L: window length / embedding dimension

    Output:
      Y: shape (n_signals, L, K)

    Raises:
      - ValueError if L is not in [2, N].
    """
    n_signals, N = y.shape
    if not (2 <= L <= N):
        raise ValueError(f"L must be in [2, N], got L={L}, N={N}")
    # sliding_window_view: (n_signals, K, L) -> transpose to (n_signals, L, K)
    Y = np.lib.stride_tricks.sliding_window_view(y, window_shape=L, axis=1)
    # Y shape: (n_signals, K, L)
    return np.transpose(Y, (0, 2, 1))


def _diagonal_averaging(Y: np.ndarray) -> np.ndarray:
    """
    Diagonal averaging (Hankelization) to reconstruct a 1D series from a trajectory matrix.

    What it does:
      - A trajectory matrix built from a 1D sequence is Hankel-structured: entries
        along anti-diagonals (constant i + j) correspond to the same time index.
      - After modifying the trajectory matrix (e.g., via SVD filtering), Hankel
        structure may be lost. Hankelization restores a 1D sequence by averaging
        each anti-diagonal.

    Input:
      Y: shape (n_signals, L, K)

    Output:
      y_hat: shape (n_signals, N), where N = L + K - 1

    Implementation details:
      - For each row i, Y[:, i, :] contributes to positions s = i .. i+K-1.
      - counts[s] stores how many elements contributed to index s, so that we can
        average properly (middle indices get more contributions than edges).
    """
    n_signals, L, K = Y.shape
    N = L + K - 1
    y_hat = np.zeros((n_signals, N), dtype=float)
    counts = np.zeros(N, dtype=float)

    # Anti-diagonals with the same index sum s = i + j
    for i in range(L):
        # row i contributes to indices s = i .. i+K-1
        y_hat[:, i : i + K] += Y[:, i, :]
        counts[i : i + K] += 1.0

    y_hat /= counts[None, :]
    return y_hat


def _svd_filter_profile(y: np.ndarray, L: int, p: int) -> np.ndarray:
    """
    SVD filtering of the integral profile y via a trajectory (Hankel) matrix.

    Steps:
      1) Build the trajectory matrix Y (embedding with window length L).
      2) Compute SVD: Y = U S V^T.
      3) Zero out the first p singular components (largest singular values).
      4) Reconstruct the filtered trajectory matrix and convert it back to a 1D
         profile with diagonal averaging (Hankelization).

    Why "zero out the first p components":
      - The leading singular components often capture strong, low-rank structure:
        trends, slow oscillations, and other dominant deterministic parts.
      - Removing them can suppress such components before running DFA, aiming to
        reduce bias in fluctuation scaling caused by large-scale deterministic structure.

    Input:
      y: shape (n_signals, N)
      L: embedding window length
      p: how many leading singular components to remove (p >= 0)

    Output:
      y_filt: shape (n_signals, N), filtered profile

    Notes:
      - SVD is computed separately for each signal.
      - p is clipped to the available rank r for each signal (p_eff = min(p, r)).
    """
    if p < 0:
        raise ValueError("p must be >= 0")
    Y = _trajectory_matrix(y, L=L)  # (n_signals, L, K)
    n_signals, L0, K = Y.shape

    # Compute SVD for each signal independently
    y_filt = np.empty_like(y)
    for idx in range(n_signals):
        Ui, si, Vti = np.linalg.svd(
            Y[idx], full_matrices=False
        )  # Ui:(L,r), si:(r,), Vti:(r,K)
        r = si.size
        p_eff = min(p, r)

        # Zero out the first p_eff singular values (largest components)
        si_f = si.copy()
        si_f[:p_eff] = 0.0

        Yi_f = (Ui * si_f[None, :]) @ Vti  # (L,K)
        y_filt[idx] = _diagonal_averaging(Yi_f[None, :, :])[0]  # back to (N,)

    return y_filt


# ---------------------------
# Helpers: DFA fluctuation F(n)
# ---------------------------


def _dfa_fluctuation_worker(y: np.ndarray, n: int, m: int) -> np.ndarray:
    """
    DFA fluctuation function for a single scale n.

    Given a (possibly filtered) profile y(t), DFA computes F(n) as:
      1) Split the profile into non-overlapping segments of length n.
         Only a whole number of segments is used (tail is discarded).
      2) In each segment, fit a polynomial trend of order m (least squares).
      3) Compute the mean squared residual in each segment.
      4) Average residual variances over segments, then take square root:
            F(n) = sqrt( mean_k( mean_t( (y_k(t) - trend_k(t))^2 ) ) )

    Input:
      y: shape (n_signals, N)
      n: segment length (scale), n >= 2
      m: polynomial order for detrending (m=1 linear, m=2 quadratic, ...)

    Output:
      F: shape (n_signals,), one F(n) value per signal

    Important constraints:
      - Need at least 2 segments to compute a meaningful average, so N // n >= 2.

    Implementation notes:
      - Uses a Vandermonde design matrix A for polynomial regression:
            A[t] = [1, t, t^2, ..., t^m]
      - Least squares is solved per segment with np.linalg.lstsq for robustness.
    """
    n_signals, N = y.shape
    if n < 2:
        raise ValueError("n must be >= 2")
    nseg = N // n
    if nseg < 2:
        raise ValueError(
            f"Too large scale n={n} for length N={N} (need at least 2 segments)"
        )

    # Cut tail so we have an integer number of segments
    y0 = y[:, : nseg * n]  # (n_signals, nseg*n)
    seg = y0.reshape(n_signals, nseg, n)  # (n_signals, nseg, n)

    t = np.arange(n, dtype=float)  # (n,)
    # Polynomial fit per segment
    F = np.empty(n_signals, dtype=float)

    for i in range(n_signals):
        # Design matrix for polynomial of order m: [1, t, t^2, ...]
        A = np.vander(t, N=m + 1, increasing=True)  # (n, m+1)

        # Compute segment-wise residual mean squares
        rms2 = np.empty(nseg, dtype=float)
        for k in range(nseg):
            coeffs, *_ = np.linalg.lstsq(A, seg[i, k], rcond=None)
            fit = A @ coeffs
            resid = seg[i, k] - fit
            rms2[k] = np.mean(resid * resid)

        # Root mean square across segments
        F[i] = np.sqrt(np.mean(rms2))

    return F


def svd_dfa(
    arr: np.ndarray,
    s: Union[int, Iterable[int]],
    *,
    integrate: bool = True,
    L: Optional[int] = None,
    p: int = 2,
    m: int = 1,
    n_min: int = 10,
    n_max: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    SVD-DFA: returns the fluctuation function F(n) (no Hurst exponent estimation inside).

    Reference (added for context on decomposition/automata link as requested):
      - Lauffer et al., 2022, "Learning Deterministic Finite Automata Decompositions from
        Examples and Demonstrations", arXiv:2205.13013.
        URL: https://arxiv.org/abs/2205.13013

    Overview:
      This routine combines an SVD-based filter on the (optional) DFA integral profile
      with classical DFA detrending to compute the scale-dependent fluctuation function F(n).

    Pipeline:
      1) (Optional) Build the integral profile y(t):
           y(t) = sum_{i<=t} (x(i) - mean(x))
         If integrate=False, the input is treated as the profile y(t) directly.
      2) SVD-filter the profile via a trajectory (Hankel) matrix:
         - embed with window length L
         - compute SVD
         - remove p leading singular components (largest singular values)
         - reconstruct the filtered profile with diagonal averaging (Hankelization)
      3) DFA: for each scale n, compute F(n) as the RMS deviation from a local polynomial trend
         of order m inside each segment of length n.

    Args:
        arr:
            1D array (N,) or 2D array (n_signals, N).
        s:
            Scales (segment lengths) n — an int or an iterable of ints.
            Example: range(10, N//4).
        integrate:
            True -> compute the DFA profile (cumsum(x-mean)).
            False -> treat arr as the profile y(t).
        L:
            Embedding window length for the SVD step.
            If None -> L = N//3 (a common heuristic).
        p:
            Number of leading singular components to remove (typically 1..3).
        m:
            Polynomial order for DFA detrending within segments (typically 1 or 2).
        n_min:
            Minimum allowed scale (small n can be unstable).
        n_max:
            Maximum allowed scale. If None -> N//4.

    Returns:
        (F_values, scales):
          - scales: shape (n_scales,)
          - F_values:
              * for a single signal: shape (n_scales,)
              * for multiple signals: shape (n_signals, n_scales)
    """
    x = _as_2d(arr)
    n_signals, N = x.shape

    if N < 50:
        raise ValueError(
            "Series is too short for DFA/SVD-DFA (need at least ~50 points)"
        )

    # scales
    if isinstance(s, int):
        scales = np.array([s], dtype=int)
    else:
        scales = np.array(list(s), dtype=int)

    if scales.size == 0:
        raise ValueError("Empty scales list")

    if n_max is None:
        n_max = N // 4
    scales = scales[(scales >= max(2, n_min)) & (scales <= n_max)]
    scales = np.unique(scales)

    if scales.size == 0:
        raise ValueError(
            f"All scales were filtered out. Try wider range, N={N}, n_max={n_max}"
        )

    # profile
    y = _integral_profile(x) if integrate else x.copy()

    # SVD params
    if L is None:
        L = max(2, N // 3)
    if not (2 <= L <= N):
        raise ValueError(f"Invalid L={L} for N={N}")

    # SVD filtering
    y_f = _svd_filter_profile(y, L=L, p=p)

    # DFA: F(n)
    F = np.empty((n_signals, scales.size), dtype=float)
    for j, n in enumerate(scales):
        F[:, j] = _dfa_fluctuation_worker(y_f, n=int(n), m=int(m))

    if arr.ndim == 1:
        return F[0], scales
    return F, scales
