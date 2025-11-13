import os

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pytest import approx

from StatTools.analysis.utils import (
    analyse_cross_ff,
    cross_fcn_sloped,
)

testdata = [
    ([5], [2, 1], [1, 1]),
    ([5, 50], [1, 2, 3], [5, 4, 1]),
]
TEST_H_VALUES = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.mark.parametrize("arguments", testdata)
def test_multiple_crossovers_utils(arguments):
    """Tests fluctuation function approximation with multiple crossovers.

    This test generates synthetic fluctuation function data with known Hurst coefficients
    and crossover points, adds noise, fits the model, and verifies that the fitted slopes
    match the true values within a relative tolerance.
    """
    C_ij, slope_ij, R_ij = arguments
    C_ij_log = list(np.log10(C_ij))
    y = [0]
    all_values = C_ij_log + slope_ij + R_ij
    tst_s = np.array(
        [0.01, 0.1, 0.3, 0.5, 1, 1.5, 2.5, 5, 7.5, 10, 15, 20, 50, 100, 250, 500, 1000]
    )
    ff = 10 ** cross_fcn_sloped(
        np.log10(tst_s),
        0,
        *all_values,
        crossover_amount=len(C_ij),
    )

    tst_hr = 1 + np.random.normal(0, 0.01, (20, len(ff)))
    tst_hr = ff * tst_hr
    ff_params_new, _ = analyse_cross_ff(tst_hr, tst_s, crossover_amount=len(C_ij))
    for i, j in zip(ff_params_new.slopes, slope_ij):

        assert i.value == pytest.approx(j, rel=0.2)
