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
    """Computes the sloped crossover function for fluctuation characteristic approximation.

    This function serves as the base element for approximating fluctuation characteristics
    with multiple Hurst coefficients, incorporating crossover points.

    Args:
        x (np.ndarray): Points where the fluctuation function F(s) is calculated.
        y_0 (float): Y-intercept of the function.
        *args: Variable length argument list containing crossover values, slopes, and rigidity parameters.
        crossover_amount (int): Number of crossover points where the Hurst coefficient changes.

    Returns:
        np.ndarray: The computed function values at the given points.
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


def get_number_parameter_by_number_crossovers(n: int) -> tuple[int, int]:
    """Returns the number of slopes and rigidity parameters for a given number of crossovers.

    Args:
        n (int): The number of crossovers in the fluctuation function.

    Returns:
        tuple[int, int]: A tuple containing the amount of slopes and the amount of rigidity parameters.
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
    ridigity_initial_parameter: float = 1,
    slope_current_initial_parameter: float = 1,
) -> tuple[ff_params, np.ndarray]:
    """Approximates the fluctuation function with multiple Hurst coefficients using non-linear least squares.

    This function fits a model with crossover points to the fluctuation function data using
    scipy.optimize.curve_fit for optimization. It returns the fitted parameters with their
    standard errors and the residuals of the fit.

    Args:
        hs (np.ndarray): The dependent data array, length M.
        S (np.ndarray): The independent variable array, shape (k, M)
        crossover_amount (int): Number of crossover points in the model.
        max_ridigity (float, optional): Maximum bound for rigidity parameters. Defaults to +np.inf.
        min_ridigity (float, optional): Minimum bound for rigidity parameters. Defaults to 1.
        min_slope_current (float, optional): Minimum bound for Hurst coefficients. Defaults to 0.
        max_slope_current (float, optional): Maximum bound for Hurst coefficients. Defaults to 5.
        ridigity_initial_parameter (float, optional): Initial guess for rigidity parameters. Defaults to 1.
        slope_current_initial_parameter (float, optional): Initial guess for Hurst coefficients. Defaults to 1.

    Returns:
        tuple[ff_params, np.ndarray]: A tuple containing the fitted parameters as an ff_params dataclass
        instance and the residuals as a numpy array.
    """

    # Initialization of optimization procedure
    s = np.repeat(S[:, np.newaxis], hs.shape[0], 1).T
    change_cross_value = partial(cross_fcn_sloped, crossover_amount=crossover_amount)
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
