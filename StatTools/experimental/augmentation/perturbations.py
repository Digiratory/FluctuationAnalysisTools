"""Module for adding perturbations to signals."""

from typing import List, Sequence, Tuple

import numpy as np
from scipy.special import gamma as gamma_func


def add_noise(signal: Sequence, ratio: float) -> np.ndarray:
    """
    Adds noise with a specified ratio of signal to noise ratio (sigma_signal / sigma_noise).

    Parameters:
        signal (Sequence): The original signal.
        ratio (float): The desired sigma_signal / sigma_noise ratio
                (for example, 10 = noise 10 times weaker).

    Returns:
        A noisy signal, noise.
    """
    signal = np.array(signal)
    sigma_signal = np.std(signal, ddof=1)
    sigma_noise = sigma_signal / ratio
    noise = np.random.normal(0, sigma_noise, size=signal.shape)
    return signal + noise, noise


def add_poisson_gaps(
    trajectory: Sequence, gap_rate: float, length_rate: float
) -> Tuple[np.ndarray, list]:
    """
    Adds gaps to the trajectory according to the Poisson flow.

    Parameters:
        trajectory (Sequence): initial trajectory.
        gap_rate (float): parameter for the Poisson flow of gaps.
        length_rate (float): parameter for the Poisson distribution of gap lengths.

    Returns:
        Tuple of two elements: the first is a trajectory_with_gaps: np.array, trajectory with gaps,
            second is a list of tuples (start, end) of missed intervals
    """
    trajectory = np.array(trajectory)
    n = len(trajectory)
    trajectory_with_gaps = trajectory.copy()
    gap_indices = []
    current_pos = 0

    while current_pos < n:
        interval = np.random.exponential(1 / gap_rate)
        current_pos += int(interval)
        if current_pos >= n:
            break
        length = np.random.poisson(length_rate)
        if length <= 0:
            length = 1
        end_pos = min(current_pos + length, n)
        trajectory_with_gaps[current_pos:end_pos] = np.nan
        gap_indices.append((current_pos, end_pos))
        current_pos = end_pos
    return trajectory_with_gaps, gap_indices


def add_exponential_gaps(
    trajectory: Sequence, rq: float, fg: float, hg: float = 0.5
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Adds gaps to the trajectory using Weibull distributions parametrized by the Hurst exponent H_G.

    Shape parameter: k = 2 - 2 * H_G.
    Scale parameters are derived so that E[X] = R_Q and E[Y] = L_avg = f_G * R_Q:
        lambda_X = R_Q / Gamma(1 + 1/k)
        lambda_Y = L_avg / Gamma(1 + 1/k)

    When H_G = 0.5, k = 1 and the Weibull reduces to the exponential distribution.

    Parameters:
        trajectory (Sequence): Initial trajectory.
        rq (float): Average interval between gap starts R_Q.
        fg (float): Desired fraction of missing values f_G (0..1).
        hg (float): Hurst exponent for gaps H_G (default 0.5 = exponential/uncorrelated).

    Returns:
        Tuple of (trajectory_with_gaps, gap_indices):
            trajectory_with_gaps — trajectory with np.nan in gap positions.
            gap_indices — list of (start, end) tuples of gap intervals.
    """
    trajectory = np.array(trajectory)
    n = len(trajectory)
    traj_with_gaps = trajectory.copy()
    gap_indices = []

    l_avg = fg * rq
    k = 2.0 - 2.0 * hg
    gamma_factor = gamma_func(1.0 + 1.0 / k)
    lambda_x = rq / gamma_factor
    lambda_y = l_avg / gamma_factor

    pos = 0
    while pos < n:
        interval = int(lambda_x * np.random.weibull(k))
        pos += interval
        if pos >= n:
            break

        length = max(1, int(lambda_y * np.random.weibull(k)))
        end_pos = min(pos + length, n)

        traj_with_gaps[pos:end_pos] = np.nan
        gap_indices.append((pos, end_pos))

        pos = end_pos

    return traj_with_gaps, gap_indices
