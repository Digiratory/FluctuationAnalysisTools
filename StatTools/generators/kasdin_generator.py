import math
import warnings
from itertools import islice
from typing import Iterator, Optional

import numpy as np
from scipy.signal import lfilter


class KasdinGenerator:
    """
    Generates a sequence of numbers according to the Kasdin model.
    Based on the method proposed in the article Kasdin, N. J. (1995).
        Discrete simulation of colored noise and stochastic processes and 1/f/sup /spl alpha// power law noise generation.
        doi:10.1109/5.381848

    Args:
        h (float): Hurst exponent (0 < H < 2) # TODO: update docs
        length (int): Maximum length of the sequence.
        random_generator (Iterator[float], optional): Iterator providing random values.
            Defaults is iter(np.random.randn(), None).
    Raises:
        ValueError: If length is less than 1
        StopIteration('Sequence exhausted') : If maximum sequence length has been reached.

    Example usage:
    >>> generator = KasdinGenerator(h, length)
    >>> trj = list(generator)
    """

    def __init__(
        self,
        h: float,
        length: int,
        random_generator: Optional[Iterator[float]] = iter(np.random.randn, None),
        normalize=True,
        filter_coefficients_length=None,
    ) -> None:
        if length is not None and length < 1:
            raise ValueError("Length must be more than 1")
        self.h = h
        self.length = length
        self.integration_count = 0
        self.random_generator = random_generator
        self.filter_coefficients_length = filter_coefficients_length

        self.needs_integration = h > 1.5
        if self.needs_integration:
            self.integration_count = math.ceil(h - 1.5)
            self.effective_h = h - self.integration_count
            beta = 2 * self.effective_h - 1
            # warnings.warn(
            #     f"Using integration step because H = {h} > 1.5. Using {self.integration_count} integration steps."
            # )
        else:
            beta = 2 * self.h - 1

        # init filter coefficients
        if self.filter_coefficients_length is None:
            self.filter_coefficients_length = self.length
            self.filter_coefficients = np.zeros(self.length, dtype=np.float64)
        else:
            self.filter_coefficients = np.zeros(
                self.filter_coefficients_length, dtype=np.float64
            )

        self.filter_coefficients[0] = 1.0
        k = np.arange(1, self.filter_coefficients_length)
        self.filter_coefficients[1:] = np.cumprod((k - 1 - beta / 2) / k)

        # generate the sequence
        random_sequence = np.fromiter(
            islice(random_generator, self.length), dtype=np.float64
        )
        self.sequence = lfilter(1, self.filter_coefficients, random_sequence)
        for _ in range(self.integration_count):
            # TODO: use scipy.integrate for accuracy
            self.sequence = np.cumsum(self.sequence)

        if np.any(np.isnan(self.sequence)) or np.any(np.isinf(self.sequence)):
            warnings.warn("Generated sequence contains invalid values.")
        if normalize:
            self.sequence -= np.mean(self.sequence)
            self.sequence /= np.std(self.sequence)
        self.current_index = 0

    def get_filter_coefficients(self) -> np.ndarray:
        """Returns the filter coefficients."""
        return self.filter_coefficients

    def __iter__(self) -> "KasdinGenerator":
        return self

    def __next__(self) -> float:
        """Return next value in sequence"""
        if self.current_index >= self.length:
            raise StopIteration("Sequence exhausted")
        self.current_index += 1
        return self.sequence[self.current_index - 1]

    def get_full_sequence(self) -> np.ndarray:
        """Return full generated sequence."""
        return self.sequence
