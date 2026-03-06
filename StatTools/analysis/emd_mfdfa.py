"""
EMD-based Multifractal Detrended Fluctuation Analysis (EMD-MFDFA)

This module implements a hybrid method combining Empirical Mode Decomposition (EMD)
with Multifractal Detrended Fluctuation Analysis (MFDFA) for analyzing complex
non-stationary time series with fractal properties.

References:
    [1] Huang, N. E., Shen, Z., Long, S. R., Wu, M. C., Shih, H. H., Zheng, Q.,
        Yen, N.-C., Tung, C. C., & Liu, H. H. (1998). The empirical mode decomposition
        and the Hilbert spectrum for nonlinear and non-stationary time series analysis.
        Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences,
        454(1971), 903-995.
        DOI: https://doi.org/10.1098/rspa.1998.0193

    [2] Kantelhardt, J. W., Zschiegner, S. A., Koscielny-Bunde, E., Havlin, S.,
        Bunde, A., & Stanley, H. E. (2002). Multifractal detrended fluctuation analysis
        of nonstationary time series. Physica A: Statistical Mechanics and its Applications,
        316(1-4), 87-114.
        DOI: https://doi.org/10.1016/S0378-4371(02)01383-3

    [3] Xi-Yuan Qian, Gao-Feng Gu, Wei-Xing Zhou (2011). Modified detrended fluctuation analysis
        based on empirical mode decomposition for the characterization of anti-persistent processes.
        Physica A: Statistical Mechanics and its Applications, 390(23–24), 4388-4395.
        DOI: https://doi.org/10.1016/j.physa.2011.07.008
"""

import warnings
from typing import Optional, Tuple, Union

import numpy as np

# PyEMD will be imported when needed to avoid hard dependency
try:
    from PyEMD import EMD

    PYEMD_AVAILABLE = True
except ImportError:
    PYEMD_AVAILABLE = False
    warnings.warn(
        "PyEMD is not installed. EMD-based MFDFA will not be available. "
        "Install it with: pip install EMD-signal"
    )


def emd_mfdfa(
    signal: np.ndarray,
    q_values: Optional[np.ndarray] = None,
    scales: Optional[np.ndarray] = None,
    degree: int = 1,
    imf_selection: str = "sum_all",
    n_integral: int = 1,
    **emd_kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform EMD-based Multifractal Detrended Fluctuation Analysis.

    This function implements a two-stage algorithm:
    1. Empirical Mode Decomposition (EMD) to decompose the signal into
       Intrinsic Mode Functions (IMFs)
    2. Multifractal Detrended Fluctuation Analysis (MFDFA) applied to
       selected IMFs

    The hybrid approach allows better handling of non-stationary signals
    by removing complex trends before fractal analysis.

    Basic usage:
        ```python
        import numpy as np
        from StatTools.analysis.emd_mfdfa import emd_mfdfa
        from StatTools.generators.kasdin_generator import create_kasdin_generator

        # Generate test signal with known Hurst exponent
        generator = create_kasdin_generator(h=0.7, length=2**12)
        signal = generator.get_full_sequence()

        # Perform EMD-based MFDFA
        q, h_q, scales = emd_mfdfa(signal, degree=2)

        # Estimate Hurst exponent at q=2
        h_estimate = h_q[np.where(q == 2)[0][0]]
        print(f"Estimated Hurst exponent: {h_estimate:.3f}")
        ```

    Args:
        signal (np.ndarray): Input time series (1D array)
        q_values (np.ndarray, optional): Array of q-orders for multifractal analysis.
            Default: np.arange(-5, 6, 1)
        scales (np.ndarray, optional): Array of time scales for analysis.
            Default: logarithmically spaced from 16 to N/4
        degree (int): Polynomial degree for detrending in MFDFA (default: 1)
        imf_selection (str): Method for selecting IMFs. Options:
            - "sum_all": Sum all IMFs (default)
            - "exclude_residual": Sum all IMFs except residual trend
            - "first_n": Use first n IMFs (requires n_imfs parameter)
            Default: "sum_all"
        n_integral (int): Number of integration operations to apply before
            MFDFA analysis (default: 1)
        **emd_kwargs: Additional keyword arguments passed to PyEMD.EMD()

    Returns:
        tuple: (q_values, h_q, scales) where:
            - q_values (np.ndarray): Array of q-order values used
            - h_q (np.ndarray): Generalized Hurst exponents for each q
            - scales (np.ndarray): Time scales used in the analysis

    Raises:
        ImportError: If PyEMD is not installed
        ValueError: If signal is too short or has wrong dimensions

    Notes:
        - Minimum recommended signal length: 2^10 (1024 points)
        - For fGn (fractional Gaussian noise): H_q ≈ constant (monofractal)
        - For fBm (fractional Brownian motion): Use n_integral=0
        - The relationship H = h(q=2) holds for standard DFA

    See Also:
        - StatTools.analysis.dfa: Standard DFA implementation
        - StatTools.analysis.fa: Fluctuation Analysis base methods
        - PyEMD documentation: https://pyemd.readthedocs.io/
    """
    # Check PyEMD availability
    if not PYEMD_AVAILABLE:
        raise ImportError(
            "PyEMD is required for EMD-based MFDFA. "
            "Install it with: pip install EMD-signal"
        )

    # Validate input
    if not isinstance(signal, np.ndarray):
        signal = np.array(signal)

    if signal.ndim != 1:
        raise ValueError("Input signal must be a 1D array")

    N = len(signal)
    if N < 64:
        raise ValueError(f"Signal too short (N={N}). Minimum recommended: 1024 points")

    if N < 1024:
        warnings.warn(
            f"Signal length (N={N}) is shorter than recommended (2^10 = 1024). "
            "Results may be unreliable."
        )

    # Set default parameters
    if q_values is None:
        q_values = np.arange(-5, 6, 1)

    if scales is None:
        s_min = 16
        s_max = N // 4
        scales = np.unique(
            np.logspace(np.log10(s_min), np.log10(s_max), num=20, dtype=int)
        )

    # STAGE 1: EMPIRICAL MODE DECOMPOSITION (EMD)

    # Initialize EMD
    emd = EMD(**emd_kwargs)

    # Decompose signal into IMFs
    # IMFs: Intrinsic Mode Functions
    # Each IMF represents oscillations at a specific time scale
    imfs = emd(signal)

    # imfs shape: (n_imfs, signal_length)
    # Last component is usually the residual trend
    n_imfs = imfs.shape[0]

    # Select IMFs based on strategy
    if imf_selection == "sum_all":
        # Sum all IMFs including residual
        preprocessed_signal = np.sum(imfs, axis=0)

    elif imf_selection == "exclude_residual":
        # Sum all IMFs except the last one (residual)
        if n_imfs > 1:
            preprocessed_signal = np.sum(imfs[:-1], axis=0)
        else:
            preprocessed_signal = imfs[0]

    elif imf_selection.startswith("first_"):
        # Extract number of IMFs to use
        try:
            n = int(imf_selection.split("_")[1])
            if n > n_imfs:
                warnings.warn(
                    f"Requested {n} IMFs but only {n_imfs} available. "
                    f"Using all {n_imfs}."
                )
                n = n_imfs
            preprocessed_signal = np.sum(imfs[:n], axis=0)
        except (IndexError, ValueError):
            raise ValueError(
                f"Invalid imf_selection: {imf_selection}. "
                "Use format 'first_N' where N is a number."
            )
    else:
        raise ValueError(
            f"Unknown imf_selection method: {imf_selection}. "
            "Valid options: 'sum_all', 'exclude_residual', 'first_N'"
        )

    # STAGE 2: MULTIFRACTAL DETRENDED FLUCTUATION ANALYSIS (MFDFA)

    # Apply MFDFA to the preprocessed signal
    h_q, Fq = _mfdfa_core(preprocessed_signal, q_values, scales, degree, n_integral)

    return q_values, h_q, scales


def _mfdfa_core(
    signal: np.ndarray,
    q_values: np.ndarray,
    scales: np.ndarray,
    degree: int,
    n_integral: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Core MFDFA algorithm implementation.

    This will be implemented based on Kantelhardt et al. (2002).

    Args:
        signal: Preprocessed signal (after EMD)
        q_values: Array of q-order values
        scales: Array of time scales
        degree: Polynomial degree for detrending
        n_integral: Number of integrations

    Returns:
        tuple: (h_q, Fq_scales) - Hurst exponents and fluctuation functions
    """
    N = len(signal)

    # Remove mean
    signal = signal - np.mean(signal)

    # Integration (profile creation)
    profile = signal
    for _ in range(n_integral):
        profile = np.cumsum(profile)

    # Initialize fluctuation function matrix
    Fq = np.zeros((len(q_values), len(scales)))

    # For each scale s
    for s_idx, s in enumerate(scales):
        # Number of segments
        n_segments = N // s

        if n_segments < 1:
            continue

        # Calculate variance in each segment
        variances = []
        for v in range(n_segments):
            # Extract segment
            segment = profile[v * s : (v + 1) * s]

            # Fit polynomial
            x = np.arange(s)
            coeffs = np.polyfit(x, segment, degree)
            fit = np.polyval(coeffs, x)

            # Calculate variance
            var = np.mean((segment - fit) ** 2)
            variances.append(var)

        variances = np.array(variances)

        # Calculate F_q(s) for each q
        for q_idx, q in enumerate(q_values):
            if q == 0:
                # Special case for q=0 (logarithmic averaging)
                Fq[q_idx, s_idx] = np.exp(0.5 * np.mean(np.log(variances + 1e-10)))
            else:
                # General case
                Fq[q_idx, s_idx] = np.power(
                    np.mean(np.power(variances, q / 2.0)), 1.0 / q
                )

    # Calculate h(q) from log-log slope
    h_q = np.zeros(len(q_values))
    for q_idx in range(len(q_values)):
        # Filter out zero or invalid values
        valid = Fq[q_idx] > 0
        if np.sum(valid) > 2:
            coeffs = np.polyfit(np.log(scales[valid]), np.log(Fq[q_idx, valid]), 1)
            h_q[q_idx] = coeffs[0]

    return h_q, Fq
