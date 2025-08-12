from sympy import (
    Function,
    Matrix,
    simplify,
    symbols,
)


def nth_order_derivative(n, k):
    x = Function("x")
    if n <= 1:
        return x(k) - x(k - 1)
    else:
        return nth_order_derivative(n - 1, k) - nth_order_derivative(n - 1, k - 1)


def get_all_coeffs(expr, n):
    x = Function("x")
    k = symbols("k")
    coeffs = {}
    for i in range(n + 1):
        term = expr.coeff(x(k - i))
        if term != 0:
            coeffs[i] = simplify(term)
    return coeffs


def Fmij_formula(n, i, j):
    k = symbols("k")
    m = n - 1

    # denominator знаменатель
    expr = nth_order_derivative(j, k)
    coeffs = get_all_coeffs(expr, j)
    den = coeffs[j]

    # numerator числитель
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


def get_sympy_filter_matrix(n):
    data = []
    for i in range(n):
        row = []
        for j in range(n):
            val = Fmij_formula(n, i, j)
            row.append(val)
        data.append(row)
    return Matrix(data)


def refine_filter_matrix(filter_matrix: Matrix, n: int, ar_filter: list):
    filter_matrix_refined = filter_matrix.copy()
    for i in range(1, n + 1):
        filter_matrix_refined = filter_matrix_refined.subs({f"a_{i}": ar_filter[i]})
    return filter_matrix_refined
