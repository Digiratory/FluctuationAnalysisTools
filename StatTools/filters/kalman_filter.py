import numpy as np
from filterpy.kalman import KalmanFilter
from numpy.typing import NDArray

from StatTools.analysis.dfa import DFA
from StatTools.filters.symbolic_kalman import (
    get_sympy_filter_matrix,
    refine_filter_matrix,
)
from StatTools.generators.kasdin_generator import KasdinGenerator


class EnhancedKalmanFilter(KalmanFilter):
    """
    Advanced Kalman filter based on filterpy.kalman.KalmanFilter
    with methods for automatic calculation of transition matrix (F)
    and measurement covariance matrix (R).
    """

    def get_R(self, signal: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Calculates the measurement covariance matrix (R) for the Kalman filter.

        Parameters:
            signal (NDArray[np.float64]): Input signal (noise)

        Returns:
            NDArray[np.float64]: A 1x1 dimension covariance matrix R
        """
        return np.std(signal) ** 2

    def _get_filter_coefficients(
        self, signal: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Helper method to get filter coefficients."""
        dfa = DFA(signal)
        h = dfa.find_h()
        generator = KasdinGenerator(h, length=signal.shape[0])
        return generator.get_filter_coefficients()

    def get_filter_matrix(self, order: int, signal: np.array, dt: float = 1.0):
        """Get the filter transition matrix based on the *.

        Parameters:
            order (int): Order of the filter.

        Returns:
           NDArray[np.float64]: Filter transition matrix.
        """
        dfa = DFA(signal)
        h = dfa.find_h()
        generator = KasdinGenerator(h, length=signal.shape[0])
        ar_filter = generator.get_filter_coefficients()
        if order == 1:
            return np.array([[1, dt], [0, 1]])
        number_matrix = refine_filter_matrix(
            get_sympy_filter_matrix(order), order, ar_filter
        )
        return np.array(number_matrix, dtype=np.float64)

    def auto_configure(
        self,
        signal: NDArray[np.float64],
        noise: NDArray[np.float64],
        dt: float = 1,
        order: int = None,
    ):
        """
        Automatically adjusts R, F based on the input data.

        Parameters:
            signal (NDArray[np.float64]): Original signal
            noise (NDArray[np.float64]): Noise signal
            dt (float): Time interval between measurements
            ar_vector(NDArray[np.float64]): Autoregressive filter coefficients
        """
        if order is None:
            order = self.dim_x
        # TODO: add Q matrix auto configuration
        self.H[0][0] = 1.0
        self.R = self.get_R(noise)
        self.F = self.get_filter_matrix(order, signal, dt)
