import numpy as np
from filterpy.kalman import KalmanFilter
from numpy.typing import NDArray

from StatTools.analysis.dfa import DFA
from StatTools.generators.kasdin_generator import KasdinGenerator


class EnhancedKalmanFilter(KalmanFilter):
    """
    Advanced Kalman filter with automatic parameter estimation.

    Extends filterpy's KalmanFilter with intelligent methods for automatically
    calculating the transition matrix (F) and measurement covariance matrix (R)
    based on signal characteristics. Uses DFA analysis and autoregressive
    modeling to estimate system dynamics from data.

    The enhanced filter can adapt to different types of signals by analyzing
    their fractal properties and long-range correlations to set appropriate
    filter parameters.

    Basic usage:
        ```python
        import numpy as np
        from StatTools.filters.kalman_filter import EnhancedKalmanFilter

        # Create enhanced Kalman filter
        kf = EnhancedKalmanFilter(dim_x=2, dim_z=1)

        # Auto-configure using signal characteristics
        kf.auto_configure(
            signal=original_signal,
            noise=noise_signal,
            dt=0.01,
            order=2
        )

        # Use like standard Kalman filter
        kf.predict()
        kf.update(measurement)
        ```

    Attributes:
        Inherits all attributes from filterpy.kalman.KalmanFilter
        Additional methods for automatic parameter estimation

    Note:
        Requires filterpy package and StatTools analysis modules.
        Automatically estimates Hurst exponent and AR coefficients from data.
    """

    def get_R(self, signal: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Calculate measurement covariance matrix (R) from signal statistics.

        Estimates the measurement noise covariance by analyzing the variance
        structure of the input signal. Uses the signal's standard deviation
        as a proxy for measurement uncertainty.

        Args:
            signal (NDArray[np.float64]): Input signal representing measurement noise

        Returns:
            NDArray[np.float64]: 1x1 covariance matrix R representing measurement uncertainty

        Note:
            Assumes measurement noise is white and Gaussian with variance equal
            to the signal's variance. For more sophisticated noise modeling,
            consider using more advanced noise estimation techniques.
        """
        return np.std(signal) ** 2

    def _get_filter_coefficients(
        self, signal: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Extract autoregressive filter coefficients from signal using DFA.

        Performs fractal analysis on the signal to estimate its Hurst exponent,
        then uses this to generate appropriate autoregressive coefficients
        for the Kalman filter transition matrix.

        Args:
            signal (NDArray[np.float64]): Input signal for coefficient estimation

        Returns:
            NDArray[np.float64]: Autoregressive filter coefficients

        Note:
            Uses DFA to estimate Hurst exponent, then KasdinGenerator to
            create appropriate AR coefficients based on the fractal properties.
        """
        dfa = DFA(signal)
        h = dfa.find_h()
        generator = KasdinGenerator(h, length=signal.shape[0])
        return generator.get_filter_coefficients()

    def get_F(
        self, signal: NDArray[np.float64], dt: float, order: int = 2
    ) -> NDArray[np.float64]:
        """
        Calculate state transition matrix (F) based on signal characteristics.

        Automatically constructs the transition matrix by analyzing the signal's
        fractal properties and estimating appropriate autoregressive coefficients.
        Supports different system orders for varying model complexity.

        Args:
            signal (NDArray[np.float64]): Input signal for analysis
            dt (float): Time step between measurements
            order (int): System order (1, 2, or 3) controlling model complexity

        Returns:
            NDArray[np.float64]: State transition matrix F

        Raises:
            ValueError: If order is not 1, 2, or 3 (unsupported orders)

        Note:
            - Order 1: Simple velocity model
            - Order 2: Acceleration model (default, most common)
            - Order 3: Jerk model (higher order dynamics)

            The matrix coefficients are derived from autoregressive parameters
            estimated from the signal's fractal properties.
        """
        dfa = DFA(signal)
        h = dfa.find_h()
        generator = KasdinGenerator(h, length=signal.shape[0])
        A = generator.get_filter_coefficients()

        if order == 1:
            # Simple position-velocity model
            return np.array([[1, dt], [0, 1]])

        if order == 2:
            # Position-velocity-acceleration model
            return np.array(
                [[-A[1] - A[2], A[2] * dt], [(-1 - A[1] - A[2]) / dt, A[2]]]
            )

        if order == 3:
            # Higher order dynamics model
            return np.array(
                [
                    [-A[1] - A[2] - A[3], A[2] + 2 * A[3], -A[3]],
                    [-1 - A[1] - A[2] - A[3], A[2] + 2 * A[3], -A[3]],
                    [-1 - A[1] - A[2] - A[3], -1 + A[2] + 2 * A[3], -A[3]],
                ]
            )

        raise ValueError(f"Order {order} is not supported")

    def auto_configure(
        self,
        signal: NDArray[np.float64],
        noise: NDArray[np.float64],
        dt: float,
        order: int = 2,
    ):
        """
        Automatically configure Kalman filter parameters from data.

        Performs complete automatic setup of the Kalman filter by analyzing
        the input signal and noise characteristics. Estimates both the
        transition matrix (F) and measurement covariance (R) from the data.

        Args:
            signal (NDArray[np.float64]): Clean reference signal for system modeling
            noise (NDArray[np.float64]): Noise signal for measurement uncertainty estimation
            dt (float): Time step between measurements
            order (int): System dynamics order (1, 2, or 3)

        Note:
            This method automatically sets:
            - F: State transition matrix (from signal analysis)
            - R: Measurement covariance matrix (from noise analysis)

            Process noise matrix Q should be set separately based on
            application requirements.

        Example:
            ```python
            kf = EnhancedKalmanFilter(dim_x=2, dim_z=1)
            kf.auto_configure(signal, noise, dt=0.01, order=2)
            kf.Q = np.eye(2) * 0.1  # Set process noise manually
            ```
        """
        # TODO: add Q matrix auto configuration
        self.R = self.get_R(noise)
        self.F = self.get_F(signal, dt, order)
