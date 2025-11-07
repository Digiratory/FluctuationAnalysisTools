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
    R_ij = [5, 4, 1]
    y = [0]
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
        np.log10(C_ij[0]),
        np.log10(C_ij[1]),
        slope_ij[0],
        slope_ij[1],
        slope_ij[2],
        R_ij[0],
        R_ij[1],
        R_ij[2],
        crossover_amount=2,
    )
    tst_hr = 1 + np.random.lognormal(0, 0.3, (20, len(ff)))
    tst_hr *= ff
    ff_params, _ = analyse_cross_ff(tst_hr, tst_s)

    assert ff_params.cross[0].value == approx(C_ij[0])
