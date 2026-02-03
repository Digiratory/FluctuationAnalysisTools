import numpy as np
from scipy.signal import fftconvolve


def chol2d_mult(X, R0):
    """
    Apply 2D Cholesky-based correlation to a multivariate signal.

    Performs multiplication of the input array by the transpose of the
    Cholesky factor of the correlation (or covariance) matrix. This is
    equivalent to imposing correlations across tracks.

    Args:
        X (ndarray):
            Input array of shape (length, n_tracks) representing independent
            signals or noise components.
        R0 (ndarray):
            Symmetric positive-definite matrix of shape (n_tracks, n_tracks)
            representing the desired correlation structure.

    Returns:
        ndarray:
            Output array of shape (length, n_tracks) with correlations imposed.
    """
    C = np.linalg.cholesky(R0)
    return X @ C.T


class MultiScaleFractionalGenerator:
    """
    Generator of multivariate fractional Gaussian-like signals with multiple
    spectral slopes (Hurst exponents). The class builds a piecewise power-law
    impulse response ensuring continuity at crossover points,
    filters white noise via FFT convolution, and optionally
    imposes cross-track correlations.

    Args:
        h_list (list[float]):
            List of Hurst exponents (spectral slopes). Must contain at least two
            values. Each value defines the slope in one spectral segment.
        crossover_points (list[int]):
            Positions where spectral slope changes. Must contain exactly
            len(h_list) - 1 elements. Each value defines the index where a new
            Hurst exponent becomes active.

    Raises:
        NotImplementedError: If the number of supplied Hurst exponents is not 2.
        AssertionError: If parameter dimensions do not match requirements.
    """

    def __init__(self, h_list, crossover_points):
        self._check_init_params(h_list, crossover_points)
        self.h_list = h_list
        self.crossover_points = crossover_points

        self._series = None
        self._pos = 0
        self.impulse_response = None

    def _check_init_params(self, h_list, crossover_points):
        """
        Validate the input parameters for Hurst exponents and crossover points.

        Args:
            h_list (list[float]):
                List of Hurst exponents.
            crossover_points (list[int]):
                List of indices marking transitions between exponents.

        Raises:
            NotImplementedError: If number of Hurst exponents is not 2.
            AssertionError: If crossover_points has incompatible length.
        """
        if len(h_list) != 2:
            raise NotImplementedError("Only two h values are supported")
        assert len(h_list) >= 2
        assert len(crossover_points) == len(h_list) - 1

    def _apply_correlation(self, signals, correlation_matrix):
        """
        Apply cross-track correlation using 2D Cholesky multiplication.

        Args:
            signals (ndarray):
                Array of shape (length, n_tracks) containing filtered signals.
            correlation_matrix (ndarray):
                Symmetric positive-definite matrix of shape (n_tracks, n_tracks)
                defining desired correlations.

        Returns:
            ndarray: The correlated multivariate signal with the same shape as signals.
        """
        signals = chol2d_mult(signals, correlation_matrix).T
        return signals

    def generate(self, length, n_tracks=2, seed=None, correlation_matrix=None):
        """
        Generate multiscale fractional signals and store them internally.

        Constructs a piecewise fractional impulse response with continuous
        matching at crossover points, filters Gaussian white noise via FFT
        convolution, and optionally applies cross-track correlations.

        Args:
            length (int):
                Output signal length.
            n_tracks (int, optional):
                Number of parallel tracks (multivariate outputs).
                Default is 2.
            seed (int or None, optional):
                Random seed for reproducibility. Default is None.
            correlation_matrix (ndarray or None, optional):
                Correlation matrix of shape (n_tracks, n_tracks). If None,
                a default matrix with 0.5 off-diagonal correlation is used.

        Returns:
            ndarray: Generated data of shape (length, n_tracks).
        """
        if seed is not None:
            np.random.seed(seed)

        if correlation_matrix is None:
            correlation_matrix = 0.5 * (
                np.ones((n_tracks, n_tracks)) - np.eye(n_tracks)
            ) + np.eye(n_tracks)

        segments = []
        prev_value = None
        edges = [1] + [c + 1 for c in self.crossover_points] + [length + 1]

        for i, h in enumerate(self.h_list):
            start = edges[-2 - i]
            end = edges[-1 - i]

            s = np.arange(start, end)
            k_i = (h - 0.5) * (s ** (h - 1.5))

            if prev_value is None:
                prev_value = k_i
            else:
                k_prev = prev_value / prev_value[0] * k_i[-1]
                segments.append(k_prev)
                prev_value = k_i
        segments = [k_i] + segments
        impulse_response = np.concatenate(segments)

        self.impulse_response = impulse_response

        signals = np.random.randn(length, n_tracks)
        for i in range(n_tracks):
            signals[:, i] = fftconvolve(signals[:, i], impulse_response, mode="same")
        signals = self._apply_correlation(signals, correlation_matrix)
        self._series = signals
        self._pos = 0

        return signals

    def __iter__(self):
        """
        Reset iterator over generated tracks.

        Returns:
            MultiScaleFractionalGenerator: Iterator over individual tracks.
        """
        self._pos = 0
        return self

    def __next__(self):
        """
        Return the next track (column) from the generated multivariate series.

        Returns:
            ndarray: A 1D array containing one generated track.

        Raises:
            ValueError: If generate() was not called before iteration.
            StopIteration: When all tracks have been returned.
        """
        if self._series is None:
            raise ValueError("Call generate() before iterating")

        if self._pos >= self._series.shape[1]:
            raise StopIteration

        value = self._series[:, self._pos]
        self._pos += 1
        return value
