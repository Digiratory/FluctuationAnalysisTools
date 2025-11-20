import numpy as np
import pytest

from StatTools.filters.symbolic_kalman import (
    get_sympy_filter_matrix,
    refine_filter_matrix,
)
from StatTools.generators.kasdin_generator import KasdinGenerator


def test_filter_auto_configure():
    """Test sympy matrix calculation for 2nd, 3rd order filters."""
    h = 0.8
    generator = KasdinGenerator(h, length=2 * 14)
    A = generator.get_filter_coefficients()
    for order in range(2, 4):
        number_matrix = refine_filter_matrix(get_sympy_filter_matrix(order), order, A)
        np_number_matrix = np.array(number_matrix)
        print(np.array([[-A[1] - A[2], A[2]], [(-1 - A[1] - A[2]), A[2]]]))
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
