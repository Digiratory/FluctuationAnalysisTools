import numpy as np
import pytest

from StatTools.filters.kalman_filter import KalmanFilter


@pytest.fixture
def kf_2x1():
    """KalmanFilter with dim_x=2, dim_z=1, constant velocity model."""
    return KalmanFilter(
        dim_x=2,
        dim_z=1,
        F=np.array([[1.0, 1.0], [0.0, 1.0]]),
        H=np.array([[1.0, 0.0]]),
        R=np.array([[1.0]]),
        Q=np.eye(2) * 0.01,
    )


class TestKalmanFilterInit:

    def test_init_shapes(self):
        """Internal attributes have correct shapes after init."""
        kf = KalmanFilter(
            dim_x=2,
            dim_z=1,
            F=np.eye(2),
            H=np.array([[1.0, 0.0]]),
            R=np.array([[1.0]]),
            Q=np.eye(2) * 0.1,
        )
        assert kf._x.shape == (2, 1)
        assert kf._P.shape == (2, 2)
        assert kf._F.shape == (2, 2)
        assert kf._H.shape == (1, 2)
        assert kf._R.shape == (1, 1)
        assert kf._Q.shape == (2, 2)

    def test_init_state_zeros(self):
        """Initial state vector is zero."""
        kf = KalmanFilter(
            dim_x=2,
            dim_z=1,
            F=np.eye(2),
            H=np.array([[1.0, 0.0]]),
            R=np.array([[1.0]]),
            Q=np.eye(2) * 0.1,
        )
        assert np.allclose(kf._x, np.zeros((2, 1)))

    def test_init_covariance_identity(self):
        """Initial covariance matrix is identity."""
        kf = KalmanFilter(
            dim_x=2,
            dim_z=1,
            F=np.eye(2),
            H=np.array([[1.0, 0.0]]),
            R=np.array([[1.0]]),
            Q=np.eye(2) * 0.1,
        )
        assert np.allclose(kf._P, np.eye(2))

    def test_init_1d_arrays_converted_to_2d(self):
        """1D input arrays are promoted to 2D via atleast_2d."""
        kf = KalmanFilter(
            dim_x=2,
            dim_z=1,
            F=np.eye(2),
            H=np.array([1.0, 0.0]),  # 1D
            R=np.array([1.0]),  # 1D
            Q=np.eye(2) * 0.1,
        )
        assert kf._H.ndim == 2
        assert kf._R.ndim == 2

    def test_init_wrong_F_shape_raises(self):
        """AssertionError raised when F has wrong shape."""
        with pytest.raises(AssertionError, match="F must be \\(2, 2\\)"):
            KalmanFilter(
                dim_x=2,
                dim_z=1,
                F=np.eye(3),
                H=np.array([[1.0, 0.0]]),
                R=np.array([[1.0]]),
                Q=np.eye(2) * 0.1,
            )

    def test_init_wrong_H_shape_raises(self):
        """AssertionError raised when H has wrong shape."""
        with pytest.raises(AssertionError, match="H must be \\(1, 2\\)"):
            KalmanFilter(
                dim_x=2,
                dim_z=1,
                F=np.eye(2),
                H=np.array([[1.0]]),
                R=np.array([[1.0]]),
                Q=np.eye(2) * 0.1,
            )

    def test_init_wrong_R_shape_raises(self):
        """AssertionError raised when R has wrong shape."""
        with pytest.raises(AssertionError, match="R must be \\(1, 1\\)"):
            KalmanFilter(
                dim_x=2,
                dim_z=1,
                F=np.eye(2),
                H=np.array([[1.0, 0.0]]),
                R=np.eye(2),
                Q=np.eye(2) * 0.1,
            )

    def test_init_wrong_Q_shape_raises(self):
        """AssertionError raised when Q has wrong shape."""
        with pytest.raises(AssertionError, match="Q must be \\(2, 2\\)"):
            KalmanFilter(
                dim_x=2,
                dim_z=1,
                F=np.eye(2),
                H=np.array([[1.0, 0.0]]),
                R=np.array([[1.0]]),
                Q=np.eye(3) * 0.1,
            )


class TestKalmanFilterPredict:

    def test_predict_position_velocity_evolution(self, kf_2x1):
        """Constant velocity model: position increases by velocity each step."""
        kf_2x1._x = np.array([[0.0], [1.0]])

        kf_2x1.predict()
        assert np.allclose(kf_2x1._x, np.array([[1.0], [1.0]]))

        kf_2x1.predict()
        assert np.allclose(kf_2x1._x, np.array([[2.0], [1.0]]))

    def test_predict_covariance_grows(self, kf_2x1):
        """Covariance increases after predict (uncertainty grows without measurement)."""
        initial_trace = np.trace(kf_2x1._P)
        kf_2x1.predict()
        assert np.trace(kf_2x1._P) > initial_trace


class TestKalmanFilterAdjust:

    def test_adjust_state_moves_toward_measurement(self, kf_2x1):
        """State moves closer to measurement after adjust."""
        kf_2x1._x = np.array([[0.0], [0.0]])
        measurement = np.array([[5.0]])

        kf_2x1.adjust(measurement)
        assert kf_2x1._x[0, 0] > 0.0

    def test_adjust_repeated_converges(self, kf_2x1):
        """Repeated adjustments with same measurement converge toward it."""
        kf_2x1._x = np.array([[0.0], [0.0]])
        measurement = np.array([[5.0]])

        kf_2x1.adjust(measurement)
        dist_first = abs(kf_2x1._x[0, 0] - 5.0)

        kf_2x1.adjust(measurement)
        assert abs(kf_2x1._x[0, 0] - 5.0) < dist_first

    def test_adjust_covariance_decreases(self, kf_2x1):
        """Covariance decreases after adjustment (measurement reduces uncertainty)."""
        initial_trace = np.trace(kf_2x1._P)
        kf_2x1.adjust(np.array([[1.0]]))
        assert np.trace(kf_2x1._P) < initial_trace

    def test_adjust_covariance_stays_symmetric(self, kf_2x1):
        """P remains symmetric after Joseph-form update."""
        for _ in range(500):
            kf_2x1.predict()
            kf_2x1.adjust(np.array([[np.random.randn()]]))
        assert np.allclose(kf_2x1._P, kf_2x1._P.T, atol=1e-10)

    def test_adjust_none_raises(self, kf_2x1):
        """ValueError raised when None is passed as measurement."""
        with pytest.raises(ValueError, match="Do not pass None as a measurement"):
            kf_2x1.adjust(None)

    def test_adjust_wrong_shape_raises(self, kf_2x1):
        """ValueError raised when measurement has wrong shape."""
        with pytest.raises(ValueError, match="Expected z shape \\(1, 1\\), got"):
            kf_2x1.adjust(np.array([[1.0], [2.0]]))


class TestKalmanFilterIntegration:

    def test_tracks_constant_position(self):
        """Filter estimate converges to true constant position."""
        kf = KalmanFilter(
            dim_x=2,
            dim_z=1,
            F=np.array([[1.0, 1.0], [0.0, 1.0]]),
            H=np.array([[1.0, 0.0]]),
            R=np.array([[1.0]]),
            Q=np.eye(2) * 0.01,
        )
        true_position = 10.0
        np.random.seed(42)
        for _ in range(50):
            z = true_position + np.random.randn()
            kf.predict()
            kf.adjust(np.array([[z]]))
        assert abs(kf._x[0, 0] - true_position) < 1.0

    def test_tracks_linear_trajectory(self):
        """Filter follows linear trajectory with low mean error."""
        kf = KalmanFilter(
            dim_x=2,
            dim_z=1,
            F=np.array([[1.0, 1.0], [0.0, 1.0]]),
            H=np.array([[1.0, 0.0]]),
            R=np.array([[1.0]]),
            Q=np.eye(2) * 0.01,
        )
        np.random.seed(42)
        errors = []
        for t in range(20):
            z = t + np.random.randn()
            kf.predict()
            kf.adjust(np.array([[z]]))
            errors.append(abs(kf._x[0, 0] - t))
        assert np.mean(errors) < 2.0

    def test_2d_measurement(self):
        """Filter works correctly with 2D measurement vector."""
        kf = KalmanFilter(
            dim_x=3,
            dim_z=2,
            F=np.eye(3),
            H=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            R=np.eye(2) * 0.5,
            Q=np.eye(3) * 0.1,
        )
        true_state = np.array([[1.0], [2.0], [3.0]])
        np.random.seed(0)
        z = kf._H @ true_state + np.random.randn(2, 1) * 0.5
        kf.predict()
        kf.adjust(z)
        assert abs(kf._x[0, 0] - 1.0) < 1.0
        assert abs(kf._x[1, 0] - 2.0) < 1.0

    def test_no_nan_or_inf_after_many_cycles(self, kf_2x1):
        """State and covariance remain finite after many predict-adjust cycles."""
        np.random.seed(7)
        for _ in range(200):
            kf_2x1.predict()
            kf_2x1.adjust(np.array([[np.random.randn()]]))
        assert not np.any(np.isnan(kf_2x1._x))
        assert not np.any(np.isinf(kf_2x1._P))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
