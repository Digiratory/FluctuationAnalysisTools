import numpy as np


class KalmanFilter:
    """Implements a Kalman filter without controlling influence.

    Attributes
    ----------
    _x : np.ndarray
        Current state estimate.
    _P : np.ndarray
        Current state covariance matrix.
    _Q : np.ndarray
        Process noise matrix.
    _F : np.ndarray
        State transition matrix.
    _H : np.ndarray
        Measurement matrix.
    _R : np.ndarray
        Measurement noise matrix.
    _I : np.ndarray
        Identity matrix of shape (dim_x, dim_x).

    Examples
    --------
    Constant-position model: 2D state (position, velocity),
    1D measurement (position only).

    >>> import numpy as np
    >>> dt = 1.0
    >>> kf = KalmanFilter(
    ...     dim_x=2,
    ...     dim_z=1,
    ...     F=np.array([[1, dt], [0, 1]]),
    ...     H=np.array([[1, 0]]),
    ...     R=np.array([[5.0]]),
    ...     Q=np.array([[0.1, 0.0], [0.0, 0.1]]),
    ... )
    >>> measurements = [1.1, 2.3, 3.0, 4.2, 5.1]
    >>> for z in measurements:
    ...     kf.predict()
    ...     kf.adjust(np.array([[z]]))
    >>> kf.get_current_state()   # [position, velocity]
    >>> kf.get_current_measurement()  # filtered position, shape (1, 1)
    """

    def __init__(
        self,
        dim_x: int,
        dim_z: int,
        F: np.ndarray,
        H: np.ndarray,
        R: np.ndarray,
        Q: np.ndarray,
    ):
        """
        Parameters
        ----------
        dim_x : int
            Filter order (dimension of the state vector).
        dim_z : int
            Dimension of the measurement vector.
        F : np.ndarray
            State transition matrix, shape (dim_x, dim_x).
        H : np.ndarray
            Measurement matrix, shape (dim_z, dim_x).
        R : np.ndarray
            Measurement noise matrix, shape (dim_z, dim_z).
        Q : np.ndarray
            Process noise matrix, shape (dim_x, dim_x).
        """
        self._Q = np.atleast_2d(Q)
        self._F = np.atleast_2d(F)
        self._H = np.atleast_2d(H)
        self._R = np.atleast_2d(R)

        assert self._F.shape == (dim_x, dim_x), f"F must be ({dim_x}, {dim_x})"
        assert self._H.shape == (dim_z, dim_x), f"H must be ({dim_z}, {dim_x})"
        assert self._R.shape == (dim_z, dim_z), f"R must be ({dim_z}, {dim_z})"
        assert self._Q.shape == (dim_x, dim_x), f"Q must be ({dim_x}, {dim_x})"
        self._x = np.zeros((dim_x, 1))
        self._P = np.eye(dim_x)
        self._I = np.eye(dim_x)

    def predict(self) -> None:
        """Extrapolation step.

        Computes the predicted system state at the next time step.
        """
        # x = Fx
        self._x = self._F @ self._x
        # P = FPF' + Q
        self._P = self._F @ self._P @ self._F.T + self._Q

    def adjust(self, z: np.ndarray) -> None:
        """Correction step.

        Adjusts the prediction to reflect the new measurement.

        Parameters
        ----------
        z : np.ndarray
            New system measurement, shape (dim_z, 1).

        Raises
        ------
        ValueError
            If None is passed instead of a measurement, or if z has wrong shape.
        """
        if z is None:
            raise ValueError("Do not pass None as a measurement")
        z = np.atleast_2d(np.asarray(z, dtype=float))
        if z.shape != (self._H.shape[0], 1):
            raise ValueError(f"Expected z shape ({self._H.shape[0]}, 1), got {z.shape}")
        # y = z - Hx
        y = z - self._H @ self._x
        # S = HPH' + R
        S = self._H @ self._P @ self._H.T + self._R
        # Si = inv(S)
        # Instead of K = PH'Si used solve
        K = np.linalg.solve(S, self._H @ self._P).T
        # x = x + Ky
        self._x = self._x + K @ y
        # Instead of P = (I - KH)P used Joseph form  P = (I - KH)P(I - KH)' + KRK'.
        # https://en.wikipedia.org/wiki/Kalman_filter#Deriving_the_posteriori_estimate_covariance_matrix
        # Bucy, Richard S., and Peter D. Joseph. Filtering for stochastic processes with applications to guidance. Vol. 326. American Mathematical Soc., 2005.
        IKH = self._I - K @ self._H
        self._P = IKH @ self._P @ IKH.T + K @ self._R @ K.T

    def get_current_state(self) -> np.ndarray:
        """Returns the current state of the system."""
        return self._x

    def get_current_measurement(self) -> np.ndarray:
        """Returns the current measurement of the system."""
        return self.get_measurement_of_state(self._x)

    def get_measurement_of_state(self, x: np.ndarray) -> np.ndarray:
        """Returns the measurement of the state."""
        return self._H @ x
