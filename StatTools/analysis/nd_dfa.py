import itertools
import multiprocessing as mp
from functools import partial
from itertools import product

import numpy as np


def generate_polynomial_basis(s, N, pd):
    """
    Generate the design matrix for polynomial fitting of total degree <= pd over an N-dimensional grid of size s.

    Parameters:
    s : int
        Size of the segment in each dimension.
    N : int
        Number of dimensions.
    pd : int
        Maximum total degree of the polynomial.

    Returns:
    X : ndarray
        Design matrix of shape (s**N, num_monomials), where num_monomials is the number of monomials of total degree <= pd.
    """
    # Create grid points: each row is a coordinate in the N-dimensional grid [0, s-1]^N
    ranges = [np.arange(s) for _ in range(N)]
    grid = np.meshgrid(*ranges, indexing="ij")
    grid_points = np.stack(grid, axis=-1).reshape(-1, N)  # Shape: (s^N, N)

    # Generate all exponent tuples (a1, a2, ..., aN) such that sum(ai) <= pd
    exponents = []
    for exp_tuple in product(range(pd + 1), repeat=N):
        if sum(exp_tuple) <= pd:
            exponents.append(exp_tuple)
    exponents = np.array(exponents)  # Shape: (num_monomials, N)
    num_monomials = exponents.shape[0]

    # Construct design matrix: each column corresponds to a monomial
    num_points = grid_points.shape[0]
    X = np.ones((num_points, num_monomials), dtype=np.float64)
    for j, exp_vec in enumerate(exponents):
        for dim in range(N):
            X[:, j] *= grid_points[:, dim] ** exp_vec[dim]
    return X


def _process_segment(args):
    """
    Worker function to process a single segment for a given scale.

    Parameters:
    args : tuple
        (start_indices, data, scale, N, n_integral, X)
        start_indices: tuple of starting indices for the segment
        data: the full mean-subtracted input array
        scale: current segment size
        N: number of dimensions
        n_integral: number of cumulative summations
        X: precomputed design matrix for polynomial fitting

    Returns:
    ssr : float
        Sum of squared residuals after polynomial detrending.
    """
    start_indices, data, scale, N, n_integral, X = args
    # Extract segment
    slices = tuple(slice(start, start + scale) for start in start_indices)
    seg_data = data[slices].copy()  # Shape: (scale, scale, ..., scale)

    # Apply cumulative summation n_integral times along each dimension
    for _ in range(n_integral):
        for axis in range(N):
            seg_data = np.cumsum(seg_data, axis=axis)

    # Flatten segment data
    y = seg_data.flatten()  # Shape: (scale**N,)

    # Polynomial detrending via least squares
    try:
        beta, residuals, rank, svals = np.linalg.lstsq(X, y, rcond=None)
        if residuals.size > 0:
            ssr = residuals[0]
        else:
            pred = X @ beta
            ssr = np.sum((y - pred) ** 2)
    except np.linalg.LinAlgError:
        ssr = np.nan
    return ssr


def nd_dfa(data, s, pd=1, n_integral=1, step=0.5, processes=1):
    """
    Perform N-dimensional Detrended Fluctuation Analysis (ND-DFA) proposed by Yuan, N. et al.[1].

    [1] Gu, Gao-Feng, and Wei-Xing Zhou. "Detrended fluctuation analysis for
        fractals and multifractals in higher dimensions." Physical Review
        E—Statistical, Nonlinear, and Soft Matter Physics 74.6 (2006).
        https://doi.org/10.1103/PhysRevE.74.061104
        https://arxiv.org/abs/physics/0607194

    Args:
        data : ndarray
            Input array of shape (d1, d2, ..., dN).
        s : int or sequence of int
            If int: maximum scale (scales generated from s_min to s with logarithmic step).
            If sequence: specific scales to evaluate.
        pd : int, optional (default=1)
            Polynomial degree for detrending (total degree).
        n_integral : int, optional (default=1)
            Number of cumulative summation operations (integration steps).
        step : float, optional (default=0.5)
            Logarithmic step size for scale generation (base 10).
        processes : int, optional (default=1)
            Number of parallel processes. Set to 1 for sequential execution.

    Returns:
        F2 : ndarray
            Detrended fluctuation values F²(s) for each valid scale.
        S : ndarray
            Valid scales used in computation (may differ from input due to filtering).

    Notes:
    - Follows the algorithm in Section II.A of Gu and Zhou (partition first, then cumulative sum).
    - Global mean subtraction is applied to input data before processing.
    - Scales are filtered to ensure: s_min <= s <= min(data.shape)//4 and s <= each dimension.
    - Polynomial fitting uses total degree <= pd in N dimensions.
    """
    data = np.asarray(data, dtype=np.float64)
    N = data.ndim

    # Subtract global mean (as implied by Eq.1 in the paper)
    data = data - np.mean(data)

    # Determine valid scales
    if isinstance(s, int):
        s_min = max(
            5, pd + 1
        )  # Minimum scale from paper (Sec. III) and polynomial constraint
        max_scale = (
            min(data.shape) // 4
        )  # Maximum scale per paper (Sec. III.A for 2D, generalized)
        if s_min > max_scale:
            raise ValueError(f"No valid scales: s_min={s_min} > max_scale={max_scale}")
        log_s_min = np.log10(s_min)
        log_s_max = np.log10(max_scale)
        num_steps = int((log_s_max - log_s_min) / step) + 1
        exponents = np.linspace(log_s_min, log_s_max, num_steps)
        candidate_scales = np.unique(np.round(10**exponents).astype(int))
    else:
        candidate_scales = np.array(s, dtype=int)

    # Filter scales: must satisfy constraints
    valid_scales = []
    for scale in candidate_scales:
        if scale < max(5, pd + 1):
            continue
        if scale > min(data.shape) // 4:
            continue
        if any(scale > dim for dim in data.shape):
            continue
        valid_scales.append(scale)
    valid_scales = sorted(set(valid_scales))
    if not valid_scales:
        raise ValueError("No valid scales after filtering constraints")

    # Prepare result containers
    F2_list = []
    S_list = []

    # Process each scale
    for scale in valid_scales:
        # Number of segments per dimension and total segments
        n_segments_per_dim = [dim // scale for dim in data.shape]
        total_segments = np.prod(n_segments_per_dim)
        if total_segments == 0:
            continue

        # Precompute design matrix for current scale and polynomial degree
        X = generate_polynomial_basis(scale, N, pd)
        num_coeffs = X.shape[1]
        num_points = scale**N

        # Skip if insufficient points for polynomial fitting
        if num_points < num_coeffs:
            continue

        # Generate segment starting indices
        starts = [np.arange(0, dim - scale + 1, scale) for dim in data.shape]
        segment_starts = list(product(*starts))
        if not segment_starts:
            continue

        # Prepare arguments for parallel processing
        worker_args = [
            (start, data, scale, N, n_integral, X) for start in segment_starts
        ]

        # Process segments in parallel or sequentially
        if processes > 1:
            with mp.Pool(processes) as pool:
                ssr_results = pool.map(_process_segment, worker_args)
        else:
            ssr_results = list(map(_process_segment, worker_args))

        # Check for computation errors
        if any(np.isnan(ssr) for ssr in ssr_results):
            continue

        # Compute F^2(s) = (1/(N_s * s^N)) * sum(ssr over segments)
        total_ssr = np.sum(ssr_results)
        F2_val = total_ssr / (total_segments * num_points)
        F2_list.append(F2_val)
        S_list.append(scale)

    if not F2_list:
        raise ValueError(
            "No scales processed successfully. Check input constraints and polynomial degree."
        )

    return np.array(F2_list), np.array(S_list)
