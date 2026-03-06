import numpy as np

from .bma import bma
from .dpcca import dpcca
from .fa import fa
from .nd_dfa import nd_dfa
from .support_ff import (
    f_fcn,
    f_fcn_without_overflaw,
    ff_base_appriximation,
    rev_f_fcn,
    tf_minus_inf,
    tf_plus_inf,
)
from .utils import (
    analyse_cross_ff,
    analyse_cross_ff_linregress,
    analyse_zero_cross_ff,
    cross_fcn_sloped,
    ff_params,
    var_estimation,
)


def analyse_ff(
    hs: np.ndarray,
    s: np.ndarray,
    crossover_amount: int,
    method: str = "linregress",
    **kwargs,
) -> tuple[ff_params, np.ndarray]:
    method = method.lower()
    if method == "relu_analyse":
        return analyse_cross_ff(hs=hs, S=s, crossover_amount=crossover_amount)
    elif method == "linregress":
        return analyse_cross_ff_linregress(
            hs=hs, s=s, crossover_amount=crossover_amount
        )
    else:
        raise ValueError(
            f"Unknown method for analyse: {method}. Available methods: 'relu_analyse', 'linregress'"
        )
