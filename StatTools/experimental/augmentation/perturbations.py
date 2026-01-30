"""Module for adding perturbations to signals."""

from typing import Sequence, Tuple

import numpy as np


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
