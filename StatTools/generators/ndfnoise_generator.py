"""Generate n-dimensional fBm field."""

import warnings

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

    Method from Dieker, T. (2004) Simulation of Fractional Brownian Motion. MSc Theses, University of Twente, Amsterdam.

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

    if isinstance(hurst, tuple):
        warnings.warn("Tuple is not implemented. Use a single float value for hurst.")
        return

    shape = tuple(shape)
    dim = len(shape)

    # Итоговая форма спектра rFFT
    spec_shape = (*shape[:-1], shape[-1] // 2 + 1)

    # Создаёт массив для хранения каждой точки спектра
    k_sq = np.zeros(spec_shape, dtype=dtype)

    # Цикл по обычным FFT-осям
    for i, n in enumerate(shape[:-1]):
        f = np.fft.fftfreq(n, d=1.0).astype(dtype)
        reshape = [1] * dim
        reshape[i] = n
        reshape[-1] = 1
        k_sq += f.reshape(reshape) ** 2

    # Последняя rFFT ось
    f_last = np.fft.rfftfreq(shape[-1], d=1.0).astype(dtype)
    k_sq += f_last.reshape((1,) * (dim - 1) + (-1,)) ** 2

    # Радиальная частота — аргумент спектра
    k = np.sqrt(k_sq, dtype=dtype)

    # Инициализация спектра
    alpha = hurst + dim / 2.0
    S = np.zeros_like(k)
    nonzero = k > 0
    S[nonzero] = k[nonzero] ** (-alpha)

    # Комплексный белый шум
    noise = (
        np.random.standard_normal(S.shape) + 1j * np.random.standard_normal(S.shape)
    ).astype(np.complex64)

    # Обратное преобразование Фурье
    field = np.fft.irfftn(noise * S, s=shape).astype(dtype)

    if normalize:
        field -= field.mean()
        std = field.std()
        if std > 0:
            field /= std

    return field
