"""
Support functions for fluctuation function approximation in crossover finder analysis.

This module provides mathematical functions to approximate fluctuation functions (F(s))
used in crossover finder analysis. The functions are designed to prevent numerical overflow
in exponential computations and support approximations for different regimes, including
positive and negative infinity cases.

Key components include:
- A numerically stable logarithm-based transformation to avoid overflow.
- ReLU-type functions for forward and reverse approximations at crossover points.
- Special handling for boundary conditions (positive and negative infinity).
- A base approximation function that combines multiple components for crossover modeling.

"""

import numpy as np


def f_fcn_without_overflaw(x: np.ndarray) -> float:
    """
    Function, when x is too large, an overflaw may be detected in in exp.
    We will carry out transformations of the logarithm in function.

    Args:
      x(np.nadarray): points where  fluctuation function F(s) is calculated.

    Returns:
      float: The return transformed values of the logarithm.
    """
    return np.maximum(0, x) + np.log1p(np.exp(-np.abs(x)))


def f_fcn(x: np.ndarray, R: int, C: int) -> float:
    """
    Function based on ReLu type function.

    Args:
      x(np.nadarray): points where  fluctuation function F(s) is calculated.
      R(int): coefficient which is proportonal to sharpness (rigidity) of a DFA crossover.
      C(int): position of the crossover.

    Returns:
      float: The return value of function with current values of x,C and R.
    """
    dx = x - C
    log_new = f_fcn_without_overflaw(R * dx)
    return (log_new / R) * dx / np.sqrt(1 + dx**2)


def rev_f_fcn(x: np.ndarray, R: int, C: int) -> float:
    """
    Reversed function based on ReLu type function.

    Args:
      x(np.nadarray): points where  fluctuation function F(s) is calculated.
      R(int): coefficient which is proportonal to sharpness (rigidity) of a DFA crossover.
      C(int): position of the crossover.

    Returns:
      float: The return value of function with current values of x,C and R.
    """
    dx = x - C
    log_new = f_fcn_without_overflaw(R * dx)
    return (log_new / R) * (-dx) / np.sqrt(1 + dx**2)


def tf_minus_inf(x: np.ndarray, R: int, C1: int) -> float:
    """
    Function when x= -inference that conditions are imposed on: in function remains rev_f_fcn part.

    Args:
        x (np.nadarray): points where  fluctuation function F(s) is calculated in (-inference).
        R (int): coefficient which is proportonal to sharpness (rigidity) of a DFA crossover.
        C1 (int): position of the crossover.

    Returns:
        float: The return value of function with current values of x,C1 and R.
    """
    return -f_fcn(x, R, C1) - rev_f_fcn(x, R, np.min(x))


def tf_plus_inf(x: np.ndarray, R: int, C2: int) -> float:
    """
    Function when x= +inference that conditions are imposed on: in function remains f_fcn part.

    Args:
        x(np.nadarray): points where fluctuation function F(s) is calculated in (+inference).
        R(int): coefficient which is proportonal to sharpness (rigidity) of a DFA crossover.
        C2(int): position of the crossover.

    Returns:
        float: The return value of function with current values of x,C2 and R.
    """
    return -rev_f_fcn(x, R, C2)


def ff_base_appriximation(
    x: np.ndarray, r_left: int, r_right: int, c_left: int, c_right: int
) -> float:
    """
    Function which can be used as base element for fluctuation characteristic approximation.

    Args:
        x(np.nadarray): points where fluctuation function F(s) is calculated in (+inference).
        r_left(int): coefficient which is proportonal to sharpness (rigidity) of a DFA crossover.
        r_right(int): coefficient which is proportonal to sharpness (rigidity) of a DFA crossover.
        c_left(int): posititon of the crossover.
        c_right(int): position of the crossover.

    Returns:
        float: The return value of function with current values of x,r_left, r_right,c_left and c_right.
    """
    if np.isinf(c_left):
        return tf_minus_inf(x, r_right, c_right)
    elif np.isinf(c_right):
        return tf_plus_inf(x, r_left, c_left)

    return f_fcn(x, r_left, c_left) + rev_f_fcn(x, r_right, c_right)
