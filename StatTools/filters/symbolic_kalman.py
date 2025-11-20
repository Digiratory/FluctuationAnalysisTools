"""Symbolic Kalman filter."""

from sympy import (
    Function,
    Matrix,
    simplify,
    symbols,
)


def nth_order_derivative(n: int, k):
    """Get the nth order derivative of x(k)."""
    x = Function("x")
    if n <= 1:
        return x(k) - x(k - 1)
    return nth_order_derivative(n - 1, k) - nth_order_derivative(n - 1, k - 1)


def get_all_coeffs(expr, n: int) -> dict:
    """Get all coefficients of the expression."""
    x = Function("x")
    k = symbols("k")
    coeffs = {}
    for i in range(n + 1):
        term = expr.coeff(x(k - i))
        if term != 0:
            coeffs[i] = simplify(term)
    return coeffs


def Fmij_formula(n: int, i: int, j: int):
    """Get the (i,j) element of the matrix."""
    k = symbols("k")
    m = n - 1

    # denominator
    expr = nth_order_derivative(j, k)
    coeffs = get_all_coeffs(expr, j)
    den = coeffs[j]

    # numerator
    a = symbols(f"a_{j + 1}")
    num = -a
    for l in range(0, m - j):
        Fik = Fmij_formula(n, i, m - l)
        expr = nth_order_derivative(m - l, k)
        C = get_all_coeffs(expr, m - l)[j]
        num -= Fik * C

    if i > j:
        expr = nth_order_derivative(i, k)
        C = get_all_coeffs(expr, i)[j + 1]
        num += C
    return simplify(num / den)


def get_sympy_filter_matrix(n: int) -> Matrix:
    """Get filter matrix."""
    data = []
    for i in range(n):
        row = []
        for j in range(n):
            val = Fmij_formula(n, i, j)
            row.append(val)
        data.append(row)
    return Matrix(data)


def refine_filter_matrix(filter_matrix: Matrix, n: int, ar_filter: list) -> Matrix:
    """Refine filter matrix using AR coefficients."""
    filter_matrix_refined = filter_matrix.copy()
    for i in range(1, n + 1):
        filter_matrix_refined = filter_matrix_refined.subs({f"a_{i}": ar_filter[i]})
    return filter_matrix_refined
