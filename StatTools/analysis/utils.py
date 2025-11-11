from dataclasses import dataclass
from functools import partial

import numpy as np
from scipy.optimize import curve_fit

from StatTools.analysis.support_ff import tf


@dataclass
class var_estimation:
    value: float
    stderr: float


@dataclass
class ff_params:
    intercept: var_estimation
    cross: list[var_estimation]
    slopes: list[var_estimation]
    ridigity: list[var_estimation]


def cross_fcn_sloped(x, y_0, *args, crossover_amount: int):
    """
    Function which can be used as base element for fluctuation characteristic approximation with several Hurst
    coefficients with levelling ...........????????????

    Args:
      x(Union[int, Iterable]): points where  fluctuation function F(s) is calculated.
      y_0(int): y-intecept for function.
      *args: Variable length argument list.
      crossover_amount(int): value of points where the Hurst coefficient has changed.

    Returns:
      float: The return value of function with current input values.
    """
    crossovers = crossover_amount
    slopes_num = crossover_amount + 1
    C = args[:crossovers]
    slope = args[crossovers : crossovers + slopes_num]
    R = args[crossovers + slopes_num : crossovers + 2 * slopes_num]

    curr_C = None
    Ridigity = None
    slope_val = None
    prev_C = -100
    result_sloped = 0
    result = np.zeros_like(x, dtype=float)

    for index in range(slopes_num):
        if index < crossovers:
            curr_C = C[index]
        else:
            curr_C = 100
            pass
        slope_val = slope[index]
        Ridigity = R[index]
        result += slope_val * tf(x, Ridigity, prev_C, curr_C)
        result_sloped += slope_val * tf(0, Ridigity, prev_C, curr_C)
        prev_C = curr_C
    return y_0 + result - result_sloped


def get_number_parameter_by_number_crossovers(n: int) -> tuple[int, int, int]:
    """
    Function that returns value of each elements of fluctuation function

    Args:
     n(int): crossovers value.

    Returns:
     tuple[int,int]: number of slopes and ridigity, respectively
    """
    slopes = n + 1
    R = n + 1
    return slopes, R


def analyse_cross_ff(
    hs: np.ndarray,
    S: np.ndarray,
    crossover_amount,
    max_ridigity: float = +np.inf,
    min_ridigity: float = 1,
    min_slope_current: float = 0,
    max_slope_current: float = 5,
    ridigity_initial_parameter: float = 5,
    slope_current_initial_parameter: float = 1,
) -> tuple[ff_params, np.ndarray]:
    """
    Function where running fluctuation characteristic approximation with several Hurst
    coefficients. It let us receive parameters of fluctuation function after approximation and errors that can be calculated
    as diagonal elements of covariation matrix.

    Args:
      hs(array): The independent (k,M) shape-array variable where data is measured.
      S(array): The dependent data M-length array.
      max_ridigity(np.ndarray): maximum value of coefficient which is proportonal to sharpness (rigidity) of a DFA crossover
      for bounds of approximation.
      min_ridigity(np.ndarray): minimum value of coefficient which is proportonal to sharpness (rigidity) of a DFA crossover
      for bounds of approximation.
      max_intercept(np.ndarray): maximum value of y-intecept for function for bounds of approximation.
      min_intercept(np.ndarray): minimum value of y-intecept for function for bounds of approximation.
      min_slope_current(np.ndarray): minimum value of Hurst coefficient for bounds of approximation.
      max_slope_current(np.ndarray): maximum value of Hurst coefficient for bounds of approximation.
      ridigity_initial_parameter(np.ndarray): initial value of coefficient which is proportonal to sharpness (rigidity) of a DFA crossover
      for non-linear least squares fitting.
      slope_current_initial_parameter(np.ndarray): initial value of Hurst coefficient for non-linear least squares fitting.
      max_ridigity(np.ndarray): maximum value of coefficient which is proportonal to sharpness (rigidity) of a DFA crossover
      for bounds of approximation for function with several Hurst coefficients.

    Returns:
        tuple[float, float, float, float,float,float,flooat,float]: [itercept, cross, slope_current, ridigity,
        intercept residuals , cross residuals, slope_current residuals, ridigity residuals], where [itercept, cross, slope_current, ridigity] - parameters
        of fluctuation function with one crossover.

        tuple[float, float, float, float, float, float, float, float]: [itercept, cross, slope_current, ridigity,
        intercept residuals , cross residuals, slope_current residuals, ridigity residuals], where [itercept, cross, slope_current, ridigity] - parameters
        of fluctuation function with several crossovers, [intercept residuals , cross residuals, slope_current residuals, ridigity residuals] - residuals
        of parameters that can be calculated as difference between parameters of function afyter fitting and parameters of
        function with dependent data for function with several crossovers.
    """

    # Initialization of optimization procedure
    # S=S
    # hs=hs
    s = np.repeat(S[:, np.newaxis], hs.shape[0], 1).T
    change_cross_value = partial(cross_fcn_sloped, crossover_amount=1)
    # po = (0, np.log10(S[len(S) // 3]), np.log10(S[2 * len(S) // 3]), 1, 1, 1, 5, 5, 5)
    s_count, r_count = get_number_parameter_by_number_crossovers(crossover_amount)

    min_ridigity = [
        min_ridigity,
    ] * r_count
    max_ridigity = [
        max_ridigity,
    ] * r_count
    min_slope_current = [
        min_slope_current,
    ] * s_count
    max_slope_current = [
        max_slope_current,
    ] * s_count
    min_crossover = [np.log10(S[0])] * crossover_amount
    max_crossover = [np.log10(S[-1])] * crossover_amount

    bounds_min = (
        [
            -np.inf,
        ]
        + min_crossover
        + min_slope_current
        + min_ridigity
    )
    bounds_max = (
        [
            1,
        ]
        + max_crossover
        + max_slope_current
        + max_ridigity
    )
    p0_crossover = [
        np.log10(S[0]) + (k + 1) * (np.log10(S[-1]) - np.log10(S[0])) / s_count
        for k in range(crossover_amount)
    ]

    po = (
        [
            0,
        ]
        + p0_crossover
        + [slope_current_initial_parameter] * s_count
        + [ridigity_initial_parameter] * r_count
    )

    # Perform an optimization
    popt, pcov, infodict, mesg, ier = curve_fit(
        change_cross_value,
        np.log10(s.flatten()),
        np.log10(hs.flatten()),
        p0=po,
        bounds=(
            bounds_min,
            bounds_max,
        ),
        full_output=True,
        maxfev=6000,
        nan_policy="raise",
    )

    # Parse results into ff_params

    stderr = np.sqrt(np.diag(pcov))

    intercept_value = popt[0]
    intercept_err = stderr[0]
    cross_values = 10 ** popt[1 : crossover_amount + 1]
    slope_values = popt[crossover_amount + 1 : 2 * crossover_amount + 2]
    slope_errs = stderr[crossover_amount + 1 : 2 * crossover_amount + 2]
    ridigity_values = popt[2 * crossover_amount + 2 :]
    ridigity_err = stderr[2 * crossover_amount + 2 :]
    cross_err = 10 ** stderr[1 : crossover_amount + 1]

    return (
        ff_params(
            intercept=var_estimation(value=intercept_value, stderr=intercept_err),
            cross=[
                var_estimation(value=v, stderr=e)
                for v, e in zip(cross_values, cross_err)
            ],
            slopes=[
                var_estimation(value=v, stderr=e)
                for v, e in zip(slope_values, slope_errs)
            ],
            ridigity=[
                var_estimation(value=v, stderr=e)
                for v, e in zip(ridigity_values, ridigity_err)
            ],
        ),
        10 ** change_cross_value(np.log10(s), *popt) - hs,
    )
