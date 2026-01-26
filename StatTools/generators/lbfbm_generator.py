import math
import warnings
from typing import Iterator, List, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.signal import lfilter


def signed_power(base: float, degree: float) -> float:
    """
    Calculates the degree of a number while preserving the sign.

    Parameters:
        base: The base of the degree
        degree: An indicator of the degree

    Returns:
        sign(base) * |base|^degree for base ≠ 0
        0 for base = 0
    """
    if base == 0:
        return 0.0
    return np.sign(base) * np.abs(base) ** degree


def normalize(data: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Normalizes the data to a zero mean and a single standard deviation.

    Parameters:
        data: Input data for normalization

    Returns:
        Normalized data array
    """
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std if std != 0 else (data - mean)


def get_adaptive_filter_coefficients(
    slices_lengths: List[int], data: np.ndarray, threshold: int = 1024
) -> List[float]:
    """
    Calculates adaptive filter coefficients by weighted segment averaging.

    Parameters:
        slices_lengths: The length of the segments to process
        data: The source data of the filter
        threshold: Length threshold for switching between averaging methods

    Returns:
        List of adaptive filter coefficients
    """
    coefficients = []
    index = 0
    for length in slices_lengths:
        if index + length > len(data):
            break

        segment = data[index : index + length]
        if length > threshold:
            weights = np.linspace(1, 0.1, length)
            coeff = np.average(segment, weights=weights)
        else:
            coeff = segment[len(segment) // 2]

        coefficients.append(coeff)
        index += length

    return coefficients


class LBFBmGenerator:
    """
    (Experimental) Generates a sequence of numbers based on the Hurst exponent.
    The Hurst exponent is a measure of long-term memory of time series.

    Args:
        h (float): Hurst exponent (0 < H < 2)
        base (int, optional): Base of the number system for bins. Defaults to 1.1.
        random_generator (Iterator[float], optional): Iterator providing random values.
            Defaults is iter(np.random.randn(), None).
        length (Optional[int], optional): Maximum length of the sequence.
            Defaults to None for unlimited sequence.

    Raises:
        ValueError: If Hurst exponent is not in a range (0, 2)
        ValueError: If base is less than 1
        ValueError: If length is less than 1
        StopIteration('Sequence exhausted') : If maximum sequence length has been reached.

    Example usage:
    >>> generator = LBFBmGenerator(h, base, length)
    >>> trj = list(generator)  # Get sequence of specified length
    """

    def __init__(
        self,
        h: float,
        base: int = 1.1,
        random_generator: Optional[Iterator[float]] = iter(np.random.randn, None),
        length: Optional[int] = 10_000,
    ) -> None:
        if not 0 < h <= 2:
            raise ValueError("Hurst exponent must be in (0, 2)")
        if base < 1:
            raise ValueError("Base must be more than 1")
        if length is not None and length < 1:
            raise ValueError("Length must be more than 1")

        self.h = h
        self.base = base
        self.current_time = 0
        self.length = length
        self.random_generator = random_generator
        self.filter_len = self._find_filter_len(base, length)

        self.bins: NDArray[np.float64] = np.zeros(self.filter_len, dtype=np.float64)
        self.bin_sizes: NDArray[np.int64] = np.array(
            [1] + [int(self.base**n) for n in range(self.filter_len - 1)],
            dtype=np.int64,
        )
        self.bin_limits: NDArray[np.int64] = np.cumsum(self.bin_sizes)
        self.max_steps: int = np.sum(self.bin_sizes)

        self._init_filter()

    def _init_filter(self) -> None:
        """Initializes the filter coefficients based on the Hurst exponent."""
        beta = 2 * self.h - 1

        # Initialize filter
        orig_len = 1
        for i in range(self.filter_len - 1):
            orig_len += int(self.base**i)

        matrix_a = np.zeros(orig_len, dtype=np.float64)
        matrix_a[0] = 1.0
        k = np.arange(1, orig_len)
        matrix_a[1:] = np.cumprod((k - 1 - beta / 2) / k)

        # Optimize filter
        self.matrix_a = get_adaptive_filter_coefficients(self.bin_sizes, matrix_a)

    def _update_bins(self, new_value: float) -> None:
        """Updates the beans with a new value."""
        updated = np.zeros_like(self.bins)
        prev = None

        for i, curr_bin in enumerate(self.bins):
            if i == 0:
                updated[i] = new_value
                continue

            if self.current_time <= self.bin_limits[i]:
                # incomplete bin
                # We do not subtract from the curr if there is no transition through bin.
                if prev is None:
                    prev = signed_power(self.bins[i - 1], (1 / self.bin_sizes[i - 1]))
                updated[i] = curr_bin + prev
                break

            curr = signed_power(curr_bin, (1 / self.bin_sizes[i]))
            if prev is None:
                prev = signed_power(self.bins[i - 1], (1 / self.bin_sizes[i - 1]))
            updated[i] = curr_bin - curr + prev
            prev = curr

        self.bins = updated

    def _find_filter_len(self, base, length):
        if base == 1:
            return length
        return int(round(math.log((length - 1) * (base - 1) + 1, base), 0) + 2)

    def _calculate_step(self) -> float:
        """Applies a filter."""
        return lfilter(1, self.matrix_a, self.bins[::-1])[-1]

    def __iter__(self) -> "LBFBmGenerator":
        return self

    def __next__(self) -> float:
        """Generates the next signal increment."""
        new_val = next(self.random_generator)
        return self.next_with_value(new_val)

    def next_with_value(self, new_val: float) -> float:
        """Generates the next signal increment using a given new value."""
        if self.length is not None and self.current_time >= self.length:
            raise StopIteration("Sequence exhausted")

        self.current_time += 1
        if self.current_time >= self.max_steps:
            warnings.warn(
                f"Sequence length {self.current_time} exceeded the maximum allowed length {self.max_steps}",
                RuntimeWarning,
            )
        self._update_bins(new_val)
        return self._calculate_step()

    def __len__(self) -> int:
        """Returns the length of the sequence if specified."""
        if self.length is None:
            raise TypeError("Length is not defined for unlimited generator")
        return self.length

    @property
    def current_bins(self) -> NDArray[np.float64]:
        """Returns the current bin values."""
        return self.bins

    def get_filter_coefficients(self) -> NDArray[np.float64]:
        """Returns the current filter coefficients."""
        return self.matrix_a

    def get_bin_sizes(self) -> NDArray[np.int64]:
        """Returns the bin sizes."""
        return self.bin_sizes

    def get_signal_from_bins(self) -> float:
        """Returns the sum of the values for each bin."""
        return np.sum(self.bins)
