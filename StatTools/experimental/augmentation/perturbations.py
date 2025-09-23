import numpy as np


def add_noise(signal: np.ndarray, ratio: float) -> np.ndarray:
    """
    Adds noise with a specified ratio of signal to noise ratio (sigma_signal / sigma_noise).

    Parameters:
        signal: The original signal.
        ratio: The desired sigma_signal / sigma_noise ratio (for example, 10 = noise 10 times weaker).

    Returns:
        A noisy signal, noise.
    """
    sigma_signal = np.std(signal, ddof=1)
    sigma_noise = sigma_signal / ratio

    noise = np.random.normal(0, sigma_noise, size=signal.shape)

    return signal + noise, noise


def add_poisson_gaps(trajectory, gap_rate, length_rate):
    """
    Adds gaps to the trajectory according to the Poisson flow.

    Parameters:
    - trajectory: np.array, initial trajectory
    - gap_rate: parameter for the Poisson flow of gaps (the more, the more frequent the gaps)
    - length_rate: parameter for the Poisson distribution of gap lengths

    Returns:
    - trajectory_with_gaps: np.array, trajectory with gaps
    - gap_indices: list of tuples (start, end) of missed intervals
    """
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
