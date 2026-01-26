"""Generate n-dimensional fBm field."""

import numpy as np


def ndfnoise(
    shape: tuple[int],
    hurst: tuple[float] | float,
    normalize: bool = True,
    dtype=np.float32,
) -> np.ndarray:
    """
    N-dimensional fractional Brownian motion (fBm) generator.
    Uses rFFT (real FFT) and float32 to reduce memory usage.

    Method from Timmer, J., & Koenig, M. (1995). On generating power law noise.
    Args:
        shape (tuple[int]): Shape of the field, example: (256, 256) or (64, 64, 64)
        hurst (tuple[float] | float): Hurst exponent H.
        normalize (bool): If True, normalize the field to have zero mean and unit variance.
                          Default is True.
        dtype (np.dtype): Data type of the field. Default is float32.

    Returns:
        field (np.ndarray): n-dimensional fractional Gaussian noise field.

    Basic usage:
        ```python
        f = ndfnoise((2048, 2048), hurst=0.8)
        ```
    """
    shape = tuple(shape)
    dim = len(shape)

    if isinstance(hurst, (tuple, list, np.ndarray)):
        if len(hurst) != dim:
            raise ValueError("hurst must have one value per dimension")
        hurst = np.asarray(hurst, dtype=dtype)

    # The final shape of the rFFT spectrum
    spec_shape = (*shape[:-1], shape[-1] // 2 + 1)

    if isinstance(hurst, np.ndarray):  # anisotropic fBm version
        S = np.ones(spec_shape, dtype=dtype)
        # Usual FFT axes
        for i, n in enumerate(shape[:-1]):
            f = np.fft.fftfreq(n, d=1.0).astype(dtype)
            reshape = [1] * dim
            reshape[i] = n
            reshape[-1] = 1
            S *= np.abs(f.reshape(reshape)) ** (-(hurst[i] + 0.5))

        # rFFT axis
        f_last = np.fft.rfftfreq(shape[-1], d=1.0).astype(dtype)
        S *= np.abs(f_last.reshape((1,) * (dim - 1) + (-1,))) ** (-(hurst[-1] + 0.5))

        # removing the singularity at zero
        S[~np.isfinite(S)] = 0.0

    else:
        # Creates an array to store each point of the spectrum
        k_sq = np.zeros(spec_shape, dtype=dtype)

        # Usual FFT axes
        for i, n in enumerate(shape[:-1]):
            f = np.fft.fftfreq(n, d=1.0).astype(dtype)
            reshape = [1] * dim
            reshape[i] = n
            reshape[-1] = 1
            k_sq += f.reshape(reshape) ** 2

        # The last rFFT axis
        f_last = np.fft.rfftfreq(shape[-1], d=1.0).astype(dtype)
        k_sq += f_last.reshape((1,) * (dim - 1) + (-1,)) ** 2

        # Radial frequency, argument of the spectrum
        k = np.sqrt(k_sq, dtype=dtype)

        # Spectrum initialization
        alpha = hurst + dim / 2.0
        S = np.zeros_like(k)
        nonzero = k > 0
        S[nonzero] = k[nonzero] ** (-alpha)

    # Complex white noise
    noise = (
        np.random.standard_normal(S.shape) + 1j * np.random.standard_normal(S.shape)
    ).astype(np.complex64)

    # The inverse Fourier transform
    field = np.fft.irfftn(noise * S, s=shape).astype(dtype)

    if normalize:
        field -= field.mean()
        std = field.std()
        if std > 0:
            field /= std
    return field
