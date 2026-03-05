"""Generate n-dimensional fBm field."""

import numpy as np


def _calculate_spectral_density(
    length: int, hurst: list[float], crossover_points: tuple[float]
) -> np.ndarray:
    """
    Calculate the piecewise power-law spectral density for multifractal noise generation.

    Args:
        length (int): Length of the signal.
        hurst (list[float]): List of Hurst exponents.
        crossover_points (tuple[float]): Crossover points defining frequency transitions.

    Returns:
        np.ndarray: Spectral density array of shape (length,).
    """
    # Calculate spectral exponents
    alpha = [2 * h + 1 for h in hurst[::-1]]

    # Generate frequency array for FFT
    freqs = np.fft.fftfreq(length, d=1.0)

    # Create spectral density with piecewise power law
    S = np.ones(length, dtype=np.float64)

    # Handle DC component
    S[0] = 0.0

    # Build piecewise spectral density
    if len(hurst) == 1:
        # Single Hurst exponent case
        S[1:] = np.abs(freqs[1:]) ** (-alpha[0])
    else:
        # Multiple Hurst exponents with crossover points
        # Convert crossover points to frequencies
        crossover_freqs = [
            1 / crossover_points[i] for i in range(len(crossover_points))
        ]

        # Start with first segment
        mask = np.abs(freqs) <= crossover_freqs[0]
        S[mask & (freqs != 0)] = np.abs(freqs[mask & (freqs != 0)]) ** (-alpha[0])

        # Add remaining segments with continuity at crossover points
        for i in range(1, len(hurst)):
            if i < len(crossover_freqs):
                # Find the crossover frequency
                cf = crossover_freqs[i]

                # Create mask for this segment
                mask = (np.abs(freqs) > crossover_freqs[i - 1]) & (np.abs(freqs) <= cf)

                # Calculate the value at the crossover point from the previous segment
                prev_cf = crossover_freqs[i - 1] if i > 0 else 0
                if prev_cf > 0:
                    # Find the spectral value at the start of this segment
                    prev_mask = np.abs(freqs) <= prev_cf
                    if np.any(prev_mask & (freqs != 0)):
                        # Get the value at the boundary
                        boundary_freq = prev_cf
                        boundary_value = boundary_freq ** (-alpha[i - 1])

                        # Apply power law for this segment, scaled to match at boundary
                        segment_freqs = np.abs(freqs[mask & (freqs != 0)])
                        S[mask & (freqs != 0)] = boundary_value * (
                            segment_freqs / boundary_freq
                        ) ** (-alpha[i])
                else:
                    # First segment after DC
                    S[mask & (freqs != 0)] = np.abs(freqs[mask & (freqs != 0)]) ** (
                        -alpha[i]
                    )
            else:
                # Last segment extends to Nyquist frequency
                mask = np.abs(freqs) > crossover_freqs[i - 1]

                # Find the spectral value at the start of this segment
                prev_cf = crossover_freqs[i - 1]
                boundary_value = prev_cf ** (-alpha[i - 1])

                # Apply power law for this segment, scaled to match at boundary
                segment_freqs = np.abs(freqs[mask & (freqs != 0)])
                S[mask & (freqs != 0)] = boundary_value * (segment_freqs / prev_cf) ** (
                    -alpha[i]
                )

    # Remove any remaining infinities or NaNs
    S[~np.isfinite(S)] = 0.0

    return S


def mfnoise(
    length: int,
    hurst: tuple[float] | float,
    crossover_points: tuple[float],
    n_tracks=1,
    normalize: bool = True,
) -> np.ndarray:
    """
    Multifractal fractional noise generator.

    Args:
        length (int): Output signal length.
        hurst (tuple[float] | float): Hurst exponent H.
        crossover_points (list[int]):
            Positions where spectral slope changes. Must contain exactly
            len(hurst) - 1 elements. Each value defines the index where a new
            Hurst exponent becomes active.
        normalize (bool): If True, normalize the field to have zero mean and unit variance.
                          Default is True.
    Returns:
        np.ndarray: Generated data of shape (n_tracks, length).

    Basic usage:
        ```python
        f = mfnoise(2**15, hurst=(0.8, 0.5), crossover_points=(200,))
        ```
    """
    # Convert hurst to list for consistent handling
    if isinstance(hurst, (int, float)):
        hurst = [hurst]
    else:
        hurst = list(hurst)

    # Validate parameters
    if len(hurst) < 1:
        raise ValueError("At least one Hurst exponent must be provided")

    if len(crossover_points) != len(hurst) - 1:
        raise ValueError(
            f"Number of crossover points ({len(crossover_points)}) must be "
            f"equal to number of Hurst exponents ({len(hurst)}) minus 1"
        )

    # Calculate spectral exponents
    alpha = [2 * h + 1 for h in hurst]
    length += 1
    # Generate frequency array for FFT
    freqs = np.fft.fftfreq(length, d=1.0)

    # Create spectral density with piecewise power law
    S = _calculate_spectral_density(length, hurst, crossover_points)

    # Generate complex white noise
    noise = np.random.standard_normal(length) + 1j * np.random.standard_normal(length)

    # Apply spectral density
    spectrum = noise * np.sqrt(S)

    # Inverse FFT to get time domain signal
    signal = np.fft.ifft(spectrum).real

    # Truncate to desired length
    signal = signal[:length]

    # Generate multiple tracks if requested
    if n_tracks > 1:
        signals = np.zeros((n_tracks, length))
        signals[0] = signal

        # Generate additional tracks with different random seeds
        for i in range(1, n_tracks):
            noise = np.random.standard_normal(length) + 1j * np.random.standard_normal(
                length
            )
            spectrum = noise * np.sqrt(S)
            signal_i = np.fft.ifft(spectrum).real[:length]
            signals[i] = signal_i
    else:
        signals = signal.reshape(1, -1)

    # Normalize if requested
    if normalize:
        for i in range(signals.shape[0]):
            mean_val = np.mean(signals[i])
            std_val = np.std(signals[i])
            if std_val > 0:
                signals[i] = (signals[i] - mean_val) / std_val
            else:
                signals[i] = signals[i] - mean_val

    # Convert to fractional noise by taking differences
    return np.diff(signals, axis=1)
