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
        h (float): Hurst exponent (0.5 < H < 1.5) # TODO: update docs
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
        self.validate_h(h)
        self.h = h
        self.length = length
        self.random_generator = random_generator
        self.filter_coefficients_length = filter_coefficients_length

        beta = self.get_beta()
        self.init_filter_coefficients(beta)

        # generate the sequence
        random_sequence = np.fromiter(
            islice(random_generator, self.length), dtype=np.float64
        )
        self.sequence = lfilter(1, self.filter_coefficients, random_sequence)
        if np.any(np.isnan(self.sequence)) or np.any(np.isinf(self.sequence)):
            warnings.warn("Generated sequence contains invalid values.")

        if normalize:
            self.sequence -= np.mean(self.sequence)
            self.sequence /= np.std(self.sequence)
        self.current_index = 0

    def get_beta(self):
        return 2 * self.h - 1

    def validate_h(self, h):
        if not 0.5 <= h <= 1.5:
            raise ValueError("H must be between 0.5 and 1.5")

    def init_filter_coefficients(self, beta):
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


class ERKasdinGenerator(KasdinGenerator):
    """Extended range version of Kasdin generator, which can be used for H < 0.5 and H > 1.5"""

    def __init__(
        self,
        h: float,
        length: int,
        random_generator: Optional[Iterator[float]] = iter(np.random.randn, None),
        normalize=True,
        filter_coefficients_length=None,
    ) -> None:
        self.effective_h = h
        self.steps_count = 0
        super().__init__(
            h, length, random_generator, normalize, filter_coefficients_length
        )

        if self.steps_count > 0:
            for _ in range(self.steps_count):
                self.sequence = np.diff(self.sequence)

        elif self.steps_count < 0:
            for _ in range(abs(self.steps_count)):
                self.sequence = np.cumsum(self.sequence)

    def validate_h(self, h):
        if not 0.5 <= h <= 1.5:
            warnings.warn(
                "H is not in range (0.5, 1.5). Using integration/differentiation."
            )

    def get_beta(self):
        if self.h > 1.5:
            while self.effective_h > 1.5:
                self.steps_count -= 1
                self.effective_h -= 1
        elif self.h < 0.5:
            while self.effective_h < 0.5:
                self.steps_count += 1
                self.effective_h += 1

        beta = 2 * self.effective_h - 1
        return beta


def create_kasdin_generator(
    h: float,
    length: int,
    random_generator: Optional[Iterator[float]] = iter(np.random.randn, None),
    normalize=True,
    filter_coefficients_length=None,
) -> KasdinGenerator | ERKasdinGenerator:
    if 0.5 <= h <= 1.5:
        return KasdinGenerator(
            h, length, random_generator, normalize, filter_coefficients_length
        )
    return ERKasdinGenerator(
        h, length, random_generator, normalize, filter_coefficients_length
    )
