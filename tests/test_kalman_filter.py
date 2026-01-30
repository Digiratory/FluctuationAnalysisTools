import numpy as np
import pytest

from StatTools.filters.kalman_filter import FractalKalmanFilter, KalmanParams
from StatTools.filters.symbolic_kalman import (
    get_sympy_filter_matrix,
    refine_filter_matrix,
)
from StatTools.generators.kasdin_generator import KasdinGenerator


def test_refine_filter_matrix():
    """Test sympy matrix calculation for 2nd, 3rd order filters."""
    h = 0.8
    generator = KasdinGenerator(h, length=2 * 14)
    A = generator.get_filter_coefficients()
    for order in range(2, 4):
        number_matrix = refine_filter_matrix(get_sympy_filter_matrix(order), order, A)
        np_number_matrix = np.array(number_matrix)
        if order == 2:
            assert np.array_equal(
                np_number_matrix,
                np.array([[-A[1] - A[2], A[2]], [(-1 - A[1] - A[2]), A[2]]]),
            )
        if order == 3:
            assert np.array_equal(
                np_number_matrix,
                np.array(
                    [
                        [-A[1] - A[2] - A[3], A[2] + 2 * A[3], -A[3]],
                        [-1 - A[1] - A[2] - A[3], A[2] + 2 * A[3], -A[3]],
                        [-1 - A[1] - A[2] - A[3], -1 + A[2] + 2 * A[3], -A[3]],
                    ]
                ),
            )


def test_set_parameters_auto_F():
    """Test that set parameters with auto F works correctly."""
    kf = FractalKalmanFilter(dim_x=2, dim_z=1)

    params = KalmanParams(
        model_h=0.5,
        noise_var=0.01,
        kasdin_length=10,
    )
    kf.set_parameters(params)
    assert kf.F.shape == (2, 2)


def test_set_parameters_invalid_F_shape():
    """Test that wrong F shape raises an error."""
    kf = FractalKalmanFilter(dim_x=2, dim_z=1)
    wrong_F = np.eye(3)

    params = KalmanParams(
        model_h=0.5,
        noise_var=0.01,
        kasdin_length=10,
        F=wrong_F,
    )

    with pytest.raises(ValueError):
        kf.set_parameters(params)


def test_init_with_cashed_F():
    """Test that set parameters with provided F works correctly."""
    kf = FractalKalmanFilter(dim_x=2, dim_z=1)
    params = KalmanParams(
        model_h=0.5, noise_var=0.01, kasdin_length=10, F=np.array([[1, 1], [1, 0]])
    )
    kf.set_parameters(params)
    assert np.equal(kf.F, np.array([[1, 1], [1, 0]])).all()
