from ctypes import c_double
from functools import partial
from math import floor
from random import gauss
from typing import Optional

from numpy import array, max, min, ndarray, uint8, zeros
from numpy.random import randn

from StatTools import StatTools_bindings
from StatTools.auxiliary import SharedBuffer


def add_h_values(vector: ndarray, k: int, h: float):
    """
    Add Hurst exponent-dependent noise to a 2D field vector.

    This function implements the midpoint displacement algorithm component
    for fractional Brownian motion generation. It adds noise scaled by
    the Hurst exponent to create the appropriate fractal properties.

    Args:
        vector (ndarray): 1D array representing a row/column of the 2D field
        k (int): Iteration level in the midpoint displacement algorithm
        h (float): Hurst exponent (0 < h < 1) controlling fractal dimension

    Returns:
        ndarray: Modified vector with added noise scaled by Hurst parameter
    """
    return array(
        [v + (pow(0.5, k * (h - 1)) * gauss(0, 1)) if v != 0 else 0 for v in vector]
    )


def quant_array(vector: ndarray, min_val: float, max_val: float):
    """
    Quantize a floating-point array to 8-bit unsigned integers.

    Performs linear quantization of the input array to the range [0, 255]
    for image representation. Values are scaled based on the provided
    minimum and maximum values.

    Args:
        vector (ndarray): Input floating-point array
        min_val (float): Minimum value for scaling
        max_val (float): Maximum value for scaling

    Returns:
        ndarray: Quantized array with dtype uint8, values in range [0, 255]
    """
    return ((vector - min_val) / (max_val - min_val) * 255).astype(uint8)


# @profile()
def fb_motion_python(h: float, field_size: int):
    """
    Generate fractional Brownian motion field using pure Python implementation.

    This function implements the midpoint displacement algorithm to generate
    2D fractional Brownian motion fields. It creates fractal surfaces with
    statistical properties controlled by the Hurst exponent.

    The algorithm works by:
    1. Initializing corner values of a grid
    2. Iteratively applying midpoint displacement
    3. Adding Hurst-dependent noise at each iteration
    4. Quantizing the result for image representation

    Args:
        h (float): Hurst exponent (0 < h < 1). Controls surface roughness:
            - h = 0.5: Random surface
            - h < 0.5: Smoother surfaces
            - h > 0.5: Rougher surfaces
        field_size (int): Determines grid size as 2^field_size + 1

    Returns:
        ndarray: 2D quantized array (uint8) representing the FBM field,
                suitable for image display

    Note:
        This is the Python implementation. For better performance, use
        fb_motion() which uses optimized C code.
    """
    n = 2**field_size + 1
    shape = n, n

    F = SharedBuffer(shape, c_double)

    F[0, 0], F[0, -1], F[-1, 0], F[-1, -1] = randn(4)
    for k in range(1, field_size + 1):
        m = 2**k

        fl = floor(n / m)

        l1 = fl
        s = fl * 2
        l2 = floor((m - 1) * n / m) + 1

        for i in range(l1, l2, s):
            for j in range(l1, l2, s):
                v1 = F[i - fl, j - fl]
                v2 = F[i - fl, j + fl]
                v3 = F[i + fl, j - fl]
                v4 = F[i + fl, j + fl]

                F[i, j] = (v1 + v2 + v3 + v4) / 4

        for i in range(0, n + 1, s):
            for j in range(fl, l2, s):
                F[i, j] = (F[i, j - fl] + F[i, j + fl]) / 2

        for j in range(0, n + 1, s):
            for i in range(fl, l2, s):
                F[i, j] = (F[i - fl, j] + F[i + fl, j]) / 2

        F.apply_in_place(func=partial(add_h_values, k=k, h=h), by_1st_dim=True)

    max_val = max(F.to_array())
    min_val = min(F.to_array())
    F.apply_in_place(
        func=partial(quant_array, min_val=min_val, max_val=max_val), by_1st_dim=True
    )

    z = array(F.to_array(), dtype=uint8)
    return z


# @profile
def fb_motion(
    h: float, field_size: int, filter_mine: Optional[ndarray] = None
) -> ndarray:
    """
    Generate fractional Brownian motion field with optimized C implementation.

    This is the main function for generating 2D fractional Brownian motion fields.
    It uses optimized C code for better performance compared to the pure Python
    implementation. The generated fields have fractal properties controlled by
    the Hurst exponent.

    Basic usage:
        ```python
        import numpy as np
        from StatTools.generators.fbm import fb_motion

        # Generate FBM field with Hurst exponent 0.7
        field = fb_motion(h=0.7, field_size=8)  # Creates 257x257 field

        # Use custom initialization array
        custom_field = np.zeros((257, 257), dtype=float)
        result = fb_motion(h=0.7, field_size=8, filter_mine=custom_field)
        ```

    Args:
        h (float): Hurst exponent (0 < h < 1) controlling surface properties:
            - h close to 0: Very rough, jagged surfaces
            - h = 0.5: Standard Brownian motion characteristics
            - h close to 1: Very smooth surfaces
        field_size (int): Determines field dimensions as (2^field_size + 1) x (2^field_size + 1).
            Larger values create bigger fields but increase computation time exponentially.
        filter_mine (ndarray, optional): Custom initialization array. If provided, the FBM
            algorithm is applied to this existing field rather than starting from random
            initialization. Must have shape (2^field_size + 1, 2^field_size + 1).

    Returns:
        ndarray: 2D quantized array (uint8) representing the fractional Brownian motion field.
                Values range from 0-255 and can be directly used for image display.

    Raises:
        ValueError: If filter_mine has incorrect shape or dimensions

    Note:
        - Performance scales with field_size as O(4^field_size)
        - Typical field_size values: 6-10 (creates 65x65 to 1025x1025 fields)
        - For very large fields, consider using parallel processing approaches

    See Also:
        fb_motion_python: Pure Python implementation (slower but more portable)
    """
    if filter_mine is None:
        n = 2**field_size + 1
        zeros_arr = zeros((n, n), dtype=float)
        StatTools_bindings.fbm_core(zeros_arr, h, field_size)
        return zeros_arr.astype(uint8)
    else:
        print("HERE")
        shape = filter_mine.shape
        if filter_mine.ndim == 1 or shape[0] != shape[1]:
            raise ValueError("Cannot process such input array!")
        if 2**field_size > shape[0]:
            raise ValueError(
                "2^degree > input array shape. You either use less or equal."
            )
        StatTools_bindings.fbm_core(filter_mine, h, field_size)
        return filter_mine.astype(uint8)


if __name__ == "__main__":
    # Example usage and performance comparison
    arr = fb_motion(0.5, 12)
