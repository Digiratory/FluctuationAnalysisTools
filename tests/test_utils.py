import numpy as np
import pytest
from numpy.testing import assert_allclose
from pytest import approx

from StatTools.analysis.utils import (
    analyse_cross_ff,
    cross_fcn_sloped,
)


def test_multiple_crossovers_utils():
    slope_ij = [1, 2, 3]
    C_ij = [5, 6]
    C_ij_log = list(np.log10(C_ij))
    R_ij = [5, 4, 1]
    y = [0]
    all_values = C_ij + slope_ij + R_ij
    tst_s = np.array(
        [0.01, 0.1, 0.3, 0.5, 1, 1.5, 2.5, 5, 7.5, 10, 15, 20, 50, 100, 250, 500, 1000]
    )
    """
    Test function which can be used as base element for fluctuation characteristic approximation with several Hurst
    coefficients with signals of known Hurst exponent.
    """
    ff = 10 ** cross_fcn_sloped(
        np.log10(tst_s),
        0,
        *all_values,
        crossover_amount=2,
    )

    # tst_h_multiple = tst_hr_multiple_approx * tst_h_multiple
    tst_hr = 1 + np.random.normal(0, 0.01, (20, len(ff)))
    tst_hr = ff * tst_hr
    ff_params_new, _ = analyse_cross_ff(tst_hr, tst_s, crossover_amount=2)
    for i, j in zip(ff_params_new.slopes, slope_ij):
        #     #  j=np.log10(j)
        #      assert i.value== pytest.approx(j)
        # np.testing.assert_allclose(np.array(ff_params_new.slopes), slope_ij, rtol=1e-5, atol=0)
        assert i == pytest.approx(j, 0.2)
