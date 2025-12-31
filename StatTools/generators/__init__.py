from itertools import islice
from typing import Optional, Union

import numpy as np

# Import the individual generators
from .kasdin_generator import create_kasdin_generator
from .lbfbm_generator import LBFBmGenerator


def generate_fbn(
    hurst: float, length: int, method: str = "kasdin", **kwargs
) -> np.ndarray:
    """Generate fractional Brownian noise using the requested method.

    Args:
        hurst: Hurst exponent of the fBn process.
        length: Number of samples to generate.
        method: Identifier of the generator (e.g. "kasdin", "lbfbm").
            Available methods:
            - "kasdin": Kasdin generator (default)
            - "lbfbm": LBFBm generator (experimental)
        **kwargs: Additional keyword arguments forwarded to the underlying generator.
            Common kwargs:
            - normalize: bool (default: True for kasdin)
            - filter_coefficients_length: int (for kasdin)
            - base: float (for lbfbm, default: 1.1)

    Returns:
        numpy.ndarray: The generated fBn time series with shape (1, length) for single series.

    Raises:
        ValueError: If method is not recognized or if generator-specific validation fails.

    Example:
        >>> import numpy as np
        >>> from StatTools.generators import generate_fbn
        >>> # Generate fBn with Hurst exponent 0.7 using default kasdin method
        >>> fbn_series = generate_fbn(hurst=0.7, length=1000)
        >>> print(fbn_series.shape)
        (1, 1000)
        >>> # Generate fBn using lbfbm method with custom parameters
        >>> fbn_series_lbfbm = generate_fbn(hurst=0.5, length=500, method="lbfbm", base=1.2)
        >>> print(fbn_series_lbfbm.shape)
        (1, 500)
    """
    method = method.lower()

    if method == "kasdin":
        # Create kasdin generator and get full sequence
        generator = create_kasdin_generator(hurst, length, **kwargs)
        sequence = generator.get_full_sequence()

        # Ensure consistent return shape (1, length)
        return sequence.reshape(1, -1)

    elif method == "lbfbm":
        # Create LBFBm generator
        # Note: LBFBmGenerator uses different parameter names and has optional length
        lbfbm_kwargs = kwargs.copy()
        if "random_generator" not in lbfbm_kwargs:
            lbfbm_kwargs["random_generator"] = iter(np.random.randn, None)

        generator = LBFBmGenerator(h=hurst, length=length, **lbfbm_kwargs)

        # Generate sequence by iterating through the generator
        sequence = np.array(list(islice(generator, length)))

        # Ensure consistent return shape (1, length)
        return sequence.reshape(1, -1)

    else:
        raise ValueError(
            f"Unknown generator method: {method}. Available methods: 'kasdin', 'lbfbm'"
        )


if __name__ == "__main__":
    pass
