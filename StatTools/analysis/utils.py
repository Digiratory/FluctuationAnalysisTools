from dataclasses import dataclass
from functools import partial

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from scipy.stats import linregress

from StatTools.analysis.support_ff import (
    f_fcn,
    ff_base_appriximation,
    rev_f_fcn,
    tf_minus_inf,
    tf_plus_inf,
)


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


def get_number_parameter_by_number_crossovers(n: int) -> tuple[int, int]:
    """Returns the number of slopes and rigidity parameters for a given number of crossovers.

    Args:
        n (int): The number of crossovers in the fluctuation function.

    Returns:
        tuple[int, int]: A tuple containing the amount of slopes and the amount of rigidity (R) parameters.
    """
    slopes = n + 1
    R = n
    return slopes, R


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
    slopes_num, _ = get_number_parameter_by_number_crossovers(crossover_amount)
    C = args[:crossovers]
    slope = args[crossovers : crossovers + slopes_num]
    R = args[crossovers + slopes_num :]

    slope_fcn = np.zeros_like(x, dtype=float)
    fcn_bias = 0

    for index in range(slopes_num):
        if index == 0:
            left_c = -np.inf
            left_r = -np.inf

        else:
            left_c = C[index - 1]
            left_r = R[index - 1]

        if index == slopes_num - 1:
            right_c = np.inf
            right_r = np.inf
        else:
            right_c = C[index]
            right_r = R[index]
        slope_val = slope[index]

        b = slope_val * ff_base_appriximation(x, left_r, right_r, left_c, right_c)
        slope_fcn += b

        fcn_bias += slope_val * ff_base_appriximation(
            0, left_r, right_r, left_c, right_c
        )

    return y_0 + slope_fcn - fcn_bias


def analyse_cross_ff(
    hs: np.ndarray,
    S: np.ndarray,
    crossover_amount,
    max_ridigity: float = np.inf,
    min_ridigity: float = 1,
    min_slope_current: float = 0.05,
    max_slope_current: float = 50,
    ridigity_initial_parameter: float = 1,
    slope_current_initial_parameter: float = 0.5,
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

    p0 = (
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
        p0=p0,
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
        10 ** change_cross_value(np.log10(s), 0, *popt) - hs,
    )


def analyse_zero_cross_ff(
    hs: np.ndarray,
    S: np.ndarray,
) -> tuple[ff_params, np.ndarray]:
    """Approximates the fluctuation function with one Hurst coefficient using linear least squares regression.

    This function fits a model with zero crossover points to the fluctuation function data. It returns the fitted parameters with their
    standard errors and the residuals of the fit.

    Args:
        hs (np.ndarray): The dependent data array, length M.
        S (np.ndarray): The independent variable array, shape (k, M).
    Returns:
        tuple[ff_params, np.ndarray]: A tuple containing the fitted parameters as an ff_params dataclass
        instance and the residuals as a numpy array.
    """

    s = np.repeat(S[np.newaxis, :], hs.shape[0], axis=0)
    log_s = np.log10(s)
    log_hs = np.log10(hs)
    x = log_s.flatten()
    y = log_hs.flatten()
    result = linregress(x, y)

    fit_model = 10 ** (result.slope * np.log10(s) + result.intercept)
    residuals = fit_model - hs
    return (
        ff_params(
            intercept=var_estimation(
                value=result.intercept, stderr=result.intercept_stderr
            ),
            cross=[],
            slopes=[var_estimation(value=result.slope, stderr=result.stderr)],
            ridigity=[],
        ),
        residuals,
    )


def ff_base_approx_synt_data(
    x: np.ndarray,
    slope1: float,
    slope2: float,
    r1: float,
    r2: float,
    c1: float,
    c2: float,
) -> np.ndarray:
    if np.isinf(c1):
        approx_data = slope2 * x
        r_contrib = 0.5 * (1 + np.tanh((x - c2) / r2))
        return r_contrib * approx_data
    if np.isinf(c2):
        approx_data = slope1 * x
        r_contrib = 0.5 * (1 - np.tanh((x - c1) / r1))
        return approx_data * r_contrib
    approx_data_left = slope1 * x
    r_contrib_left = 0.5 * (1 - np.tanh((x - c1) / r1))
    approx_data_right = slope2 * x
    r_contrib_right = 0.5 * (1 + np.tanh((x - c2) / r2))
    return (
        approx_data_left * r_contrib_left
        + (1 - r_contrib_left) * approx_data_right * r_contrib_right
    )


def cross_fcn_sloped_synt_data(x, y_0, *args, crossover_amount: int):
    crossovers = crossover_amount
    slopes_num, r_count = get_number_parameter_by_number_crossovers(crossover_amount)
    c = args[:crossovers]
    slope = args[crossovers : crossovers + slopes_num]
    r = args[crossovers + slopes_num :]
    res = np.zeros_like(x, dtype=float)
    if crossovers == 0:
        res = slope[0] * x
    else:
        res = slope[0] * x
        for i in range(crossovers):
            r_contrib = 0.5 * (1 + np.tanh((x - c[i]) / r[i]))
            res += r_contrib * (slope[i + 1] - slope[i]) * x
    if len(x) > 0:
        bias = res[0] - y_0
        res = res - bias  # Normalization for initial values come from 0
    return res


def ff_base_approximation_linregress(
    x: np.ndarray,
    r1: float,
    r2: float,
    c1: float,
    c2: float,
    s: np.ndarray,
    hs: np.ndarray,
) -> np.ndarray:
    log_s = np.log10(s)
    log_hs = np.log10(hs)
    if np.isinf(c1):
        cross_index = np.searchsorted(log_s, c2)
        cross_index = max(
            4, min(cross_index, len(log_s) - 4)
        )  # restrictions for crossover location
        res = stats.linregress(log_s[cross_index:], log_hs[cross_index:])
        approx_data = res.slope * x + res.intercept
        r_contrib = 0.5 * (1 + np.tanh((x - c2) / r2))
        return r_contrib * approx_data
    if np.isinf(c2):
        cross_index = np.searchsorted(log_s, c1)
        cross_index = max(
            4, min(cross_index, len(log_s) - 4)
        )  # restrictions for crossover location
        res = stats.linregress(log_s[:cross_index], log_hs[:cross_index])
        approx_data = res.slope * x + res.intercept
        r_contrib = 0.5 * (1 - np.tanh((x - c1) / r1))
        return r_contrib * approx_data

    cross_index_1 = np.searchsorted(log_s, c1)
    cross_index_1 = max(4, min(cross_index_1, len(log_s) - 4))
    res_left = stats.linregress(log_s[:cross_index_1], log_hs[:cross_index_1])
    approx_data_left = res_left.slope * x + res_left.intercept
    r_contrib_left = 0.5 * (1 - np.tanh((x - c1) / r1))

    cross_index_2 = np.searchsorted(log_s, c2)
    cross_index_2 = max(4, min(cross_index_2, len(log_s) - 4))
    res_right = stats.linregress(log_s[cross_index_2:], log_hs[cross_index_2:])
    approx_data_right = res_right.slope * x + res_right.intercept
    r_contrib_right = 0.5 * (1 - np.tanh((x - c2) / r2))

    return (
        r_contrib_left * approx_data_left
        + (1 - r_contrib_left) * r_contrib_right * approx_data_right
    )


def cross_fcn_sloped_linregress(
    x, y_0, *args, crossover_amount: int, s: np.ndarray = None, hs: np.ndarray = None
):
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
    slopes_num, _ = get_number_parameter_by_number_crossovers(crossover_amount)
    C = args[:crossovers]
    slope = args[crossovers : crossovers + slopes_num]
    R = args[crossovers + slopes_num :]

    slope_fcn = np.zeros_like(x, dtype=float)
    fcn_bias = 0

    for index in range(slopes_num):
        if index == 0:
            left_c = -np.inf
            left_r = R[0] if len(R) > 0 else 1

        else:
            left_c = C[index - 1]
            left_r = R[index - 1]

        if index == slopes_num - 1:
            right_c = np.inf
            right_r = 1
        else:
            right_c = C[index]
            right_r = R[index]
        slope_val = slope[index]

        b = ff_base_approximation_linregress(
            x, left_r, right_r, left_c, right_c, s=s, hs=hs
        )
        slope_fcn += b

        fcn_bias += slope_val * ff_base_approximation_linregress(
            0, left_r, right_r, left_c, right_c, s=s, hs=hs
        )

    return y_0 + slope_fcn - fcn_bias


def find_optimal_cross(
    hs, s, crossover_amount=1, min_cross_data_restr=0.2, max_cross_data_restr=0.8
):
    log_s = np.log10(s)
    log_hs = np.log10(hs.flatten())
    cross_points = len(s)
    min_idx_cross = max(4, int(cross_points * min_cross_data_restr))
    max_idx_cross = min(len(s) - 4, int(cross_points * max_cross_data_restr))

    if crossover_amount == 1:
        errors = []
        cross_vars = []
        for cross_idx in range(min_idx_cross, max_idx_cross):
            res_left = stats.linregress(log_s[:cross_idx], log_hs[:cross_idx])
            res_right = stats.linregress(log_s[cross_idx:], log_hs[cross_idx:])
            left_pred = res_left.slope * log_s[:cross_idx] + res_left.intercept
            right_pred = res_right.slope * log_s[cross_idx:] + res_right.intercept
            err_left = np.sum((log_hs[:cross_idx] - left_pred) ** 2)
            err_right = np.sum((log_hs[cross_idx:] - right_pred) ** 2)
            err = err_left + err_right
            errors.append(err)
            cross_vars.append(cross_idx)

        best_idx_error = np.argmin(errors)
        best_cross_idx = cross_vars[best_idx_error]
        best_cross_S = s[best_cross_idx]
        res_left = stats.linregress(log_s[:best_cross_idx], log_hs[:best_cross_idx])
        res_right = stats.linregress(log_s[best_cross_idx:], log_hs[best_cross_idx:])
        r_diff = np.abs(
            (res_right.slope * log_s + res_right.intercept)
            - (res_left.slope * log_s + res_left.intercept)
        )
        r_bounds = 0.16 * np.max(r_diff)
        transit = r_diff < r_bounds
        transit_idx = np.where(transit)[0]
        if len(transit_idx) > 1:
            r_estimate = (log_s[transit_idx[-1]] - log_s[transit_idx[0]]) / 2
            r_estimate = max(0.1, min(r_estimate, 2))
        else:
            r_estimate = 0.5

        return {
            "cross": [best_cross_S],
            "slopes": [float(res_left.slope), float(res_right.slope)],
            "intercept": float(res_left.intercept),
            "r": [float(r_estimate)],
        }

    else:
        crossover_points = np.linspace(
            min_idx_cross, max_idx_cross, crossover_amount + 2
        ).astype(int)[1:-1]
        crossovers = [s[index] for index in crossover_points]
        slopes = []
        r = [0.5] * crossover_amount
        res_first_value = stats.linregress(
            log_s[: crossover_points[0]], log_hs[: crossover_points[0]]
        )
        slopes.append(float(res_first_value.slope))
        intercept = float(res_first_value.intercept)

        for i in range(len(crossover_points) - 1):
            res_value = stats.linregress(
                log_s[crossover_points[i] : crossover_points[i + 1]],
                log_hs[crossover_points[i] : crossover_points[i + 1]],
            )
            slopes.append(float(res_value.slope))

        res_last_value = stats.linregress(
            log_s[crossover_points[-1] :], log_hs[crossover_points[-1] :]
        )
        slopes.append(float(res_last_value.slope))

        return {"cross": crossovers, "slopes": slopes, "intercept": intercept, "r": r}


def analyse_cross_ff_linregress(
    hs: np.ndarray,
    S: np.ndarray,
    crossover_amount,
    max_ridigity: float = 2,
    min_ridigity: float = 0.1,
    min_slope_current: float = 0.01,
    max_slope_current: float = 5,
    ridigity_initial_parameter: float = 0.5,
    slope_current_initial_parameter: float = 0.5,
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
    if hs.ndim == 1:
        hs = hs.reshape(1, -1)
    if crossover_amount > 1:
        raise ValueError("values of lags can't be more then 1")
    s = np.repeat(S[:, np.newaxis], hs.shape[0], 1).T
    change_cross_value = partial(
        cross_fcn_sloped_linregress, crossover_amount=crossover_amount
    )
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
            +np.inf,
        ]
        + max_crossover
        + max_slope_current
        + max_ridigity
    )
    p0_crossover = [
        np.log10(S[0]) + (k + 1) * (np.log10(S[-1]) - np.log10(S[0])) / s_count
        for k in range(crossover_amount)
    ]

    init_params = find_optimal_cross(hs, S, crossover_amount=crossover_amount)
    p0 = (
        [init_params["intercept"]]
        + [np.log10(c) for c in init_params["cross"]]
        + init_params["slopes"]
        + init_params["r"]
    )

    def fit_func(x, *init_params):
        return cross_fcn_sloped_linregress(
            x,
            init_params[0],
            *init_params[1:],
            crossover_amount=crossover_amount,
            s=S,
            hs=hs.flatten()
        )

    # Perform an optimization
    popt, pcov, infodict, mesg, ier = curve_fit(
        fit_func,
        np.log10(s.flatten()),
        np.log10(hs.flatten()),
        p0=p0,
        bounds=(bounds_min, bounds_max),
        full_output=True,
        maxfev=10000,
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
        10 ** fit_func(np.log10(s), *popt) - hs,
    )
