import numpy as np


def f_fcn_without_ooverflaw(x):
    """
    Function, when x is too large, an overflaw may be detected in in exp.
    We will carry out transformations of the logarithm in function.

    Args:
      x(np.nadarray): points where  fluctuation function F(s) is calculated.

    Returns:
      float: The return transformed values of the logarithm.
    """
    return np.maximum(0, x) + np.log1p(np.exp(-np.abs(x)))


def f_fcn(x, R, C):
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
    log_new = f_fcn_without_ooverflaw(R * dx)
    return (log_new / R) * dx / np.sqrt(1 + dx**2)


def rev_f_fcn(x, R, C):
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
    log_new = f_fcn_without_ooverflaw(R * dx)
    return (log_new / R) * (-dx) / np.sqrt(1 + dx**2)


def tf_minus_inf(x, R, C1):
    """
    Function when x= -inference that conditions are imposed on: in function remains rev_f_fcn part.

    Args:
        x (np.nadarray): points where  fluctuation function F(s) is calculated in (-inference).
        R (int): coefficient which is proportonal to sharpness (rigidity) of a DFA crossover.
        C1 (int): position of the crossover.

    Returns:
        float: The return value of function with current values of x,C1 and R.
    """
    return rev_f_fcn(x, R, C1)


def tf_plus_inf(x, R, C2):
    """
    Function when x= +inference that conditions are imposed on: in function remains f_fcn part.

    Args:
        x(np.nadarray): points where fluctuation function F(s) is calculated in (+inference).
        R(int): coefficient which is proportonal to sharpness (rigidity) of a DFA crossover.
        C2(int): position of the crossover.

    Returns:
        float: The return value of function with current values of x,C2 and R.
    """
    return f_fcn(x, R, C2)


def tf(x, R, C1, C2):
    """
    Function which can be used as base element for fluctuation characteristic approximation.

    Args:
        x(np.nadarray): points where fluctuation function F(s) is calculated in (+inference).
        R(int): coefficient which is proportonal to sharpness (rigidity) of a DFA crossover.
        C1(int): posititon of the crossover.
        C2(int): position of the crossover.

    Returns:
        float: The return value of function with current values of x,C1,C2 and R.
    """
    if C1 <= float("-inf"):
        return tf_minus_inf(x, R, C1)
    elif C2 >= float("inf"):
        return tf_plus_inf(x, R, C2)
    else:
        return -f_fcn(x, R, C2) - rev_f_fcn(x, R, C1)
