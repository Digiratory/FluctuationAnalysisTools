from numpy import append, array, convolve, insert, mean, ndarray, ones


def movmean(arr: ndarray, k: int) -> ndarray:
    """
    Compute moving average (moving mean) of a 1D array.

    Applies a moving average filter to smooth the input signal by computing
    the average of neighboring values within a sliding window of size k.
    The function handles edge effects by extending the window symmetrically
    at the boundaries.

    Basic usage:
        ```python
        import numpy as np
        from StatTools.analysis.movmean import movmean

        # Create sample data
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        # Apply moving average with window size 3
        smoothed = movmean(data, k=3)
        # Result: [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.5, 9.5]
        ```

    Args:
        arr (ndarray): Input 1D array of numerical values
        k (int): Window size for moving average. Must be positive and
            <= length of input array.

    Returns:
        ndarray: Smoothed array with same length as input. Edge values are
            computed using progressively smaller windows to maintain output length.

    Raises:
        ValueError: If window size k > length of input array

    Note:
        - For edge handling, the function uses symmetric padding with progressively
          smaller windows near the boundaries
        - The output maintains the same length as the input array
        - Window size should be chosen based on the desired smoothing level
    """
    if k > len(arr):
        raise ValueError("Window size > len(input_array)!")
    conv = convolve(arr, ones((k,)) / k, mode="valid")
    if k == len(conv):
        return conv
    i = k // 2 + 1
    l = len(arr)
    while True:
        conv = insert(conv, 0, mean(arr[0:i]))
        if len(conv) == l:
            break
        conv = append(conv, mean(arr[-i:]))
        if len(conv) == l:
            break
        i -= 1
    return conv


if __name__ == "__main__":
    A = array([4, 8, 6, -1, -2, -3, -1, 3, 4, 5])
    # A = normal(10, 3, 1000)
    m = movmean(A, 11)
    print(m, len(m))
