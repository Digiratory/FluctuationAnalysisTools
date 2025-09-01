import operator
from contextlib import closing
from ctypes import c_double, c_int64
from functools import partial, reduce
from multiprocessing import Array, Lock, Pool, Value, cpu_count
from operator import mul
from threading import Thread
from typing import Union

import numpy
from numpy import array, copyto, frombuffer, ndarray, s_


class SharedBuffer:
    """
    Shared memory buffer for efficient inter-process data sharing.

    This class provides a convenient interface for managing shared memory arrays
    across multiple processes. It supports 1D, 2D, and 3D numerical arrays and
    includes methods for data manipulation and parallel processing.

    The SharedBuffer uses multiprocessing.Array for memory sharing, allowing
    efficient data exchange between processes without expensive serialization.

    Basic usage:
        ```python
        import numpy as np
        from StatTools.auxiliary import SharedBuffer

        # Create shared buffer for 1000x1000 float array
        shape = (1000, 1000)
        buffer = SharedBuffer(shape, c_double)

        # Copy data to shared memory
        data = np.random.normal(0, 1, shape)
        buffer.write(data)

        # Access data from shared memory
        shared_data = buffer.to_array()

        # Apply function to entire buffer
        buffer.apply_in_place(np.sin)
        ```

    Args:
        shape (tuple): Shape of the array (1D, 2D, or 3D)
        dtype: Data type for the array (c_double, c_int64, etc.)

    Attributes:
        shape (tuple): Array shape
        dtype: Array data type
        buffer (Array): Underlying multiprocessing shared memory array

    Raises:
        NotImplementedError: If array dimension > 3

    Note:
        - Supports only numerical data types
        - Memory is automatically managed and cleaned up
        - Thread-safe for read/write operations
    """

    def __init__(self, shape: tuple, dtype=Union[c_double, c_int64]):
        if len(shape) > 3:
            raise NotImplementedError("Only 1d, 2d- matrices are supported for now!")

        self.dtype, self.shape = dtype, shape
        self.offset = shape[1] if len(shape) == 2 else 1
        self.buffer = Array(dtype, reduce(mul, self.shape), lock=True)
        self.iter_counter = 0

    def __getitem__(self, item):
        """
        Get item from shared buffer using numpy-style indexing.

        Args:
            item: Index or slice object

        Returns:
            ndarray or scalar: Requested data from shared memory
        """
        if isinstance(item, int):
            return self.__get_handle()[item]
        else:
            return self.__get_handle()[s_[item]]

    def __setitem__(self, key, value):
        """
        Set item in shared buffer using numpy-style indexing.

        Args:
            key: Index or slice object
            value: Value to set
        """
        if isinstance(key, int):
            self.__get_handle()[key] = value
        else:
            self.__get_handle()[s_[key]] = value

    def __repr__(self):
        """String representation of the shared buffer."""
        return str(self.__get_handle())

    def __iter__(self):
        """Iterator over the first dimension of the array."""
        return self

    def __next__(self):
        """Get next item in iteration."""
        while self.iter_counter < self.shape[0]:
            self.iter_counter += 1
            return self[self.iter_counter - 1]
        self.iter_counter = 0
        raise StopIteration

    def __del__(self):
        """Clean up shared memory resources."""
        del self.buffer

    def __get_handle(self):
        """Get numpy array handle to shared memory."""
        return frombuffer(self.buffer.get_obj(), dtype=self.dtype).reshape(self.shape)

    def write(self, arr: ndarray, by_1_st_dim: bool = False) -> None:
        """
        Write numpy array to shared memory buffer.

        Args:
            arr (ndarray): Source array to copy
            by_1_st_dim (bool): If True, copy row-by-row for 2D arrays

        Raises:
            ValueError: If array shapes don't match
        """
        if arr.shape != self.shape:
            raise ValueError(f"Input array must have the same shape! arr: {arr.shape}")

        if by_1_st_dim:
            for i, v in enumerate(self.__get_handle()):
                v[:] = arr[i]
        else:
            copyto(self.__get_handle(), arr)

    def apply(self, func, by_1st_dim=False):
        """
        Apply function to buffer data without modifying original.

        Args:
            func: Function to apply
            by_1st_dim (bool): Apply to each row if True, entire array if False

        Returns:
            Result of function application
        """
        result = []
        if by_1st_dim:
            for i, v in enumerate(self):
                result.append(func(v))
            return result
        else:
            return func(self.__get_handle().reshape(self.shape))

    # @profile
    def apply_in_place(self, func, by_1st_dim=False):
        """
        Apply function to buffer data in-place.

        Args:
            func: Function to apply
            by_1st_dim (bool): Apply to each row if True, entire array if False
        """
        if by_1st_dim:
            for i, v in enumerate(self):
                self[i] = func(v)
        else:
            self.to_array()[:] = func(self.to_array())

    def to_array(self):
        """Convert shared buffer to regular numpy array."""
        return self.__get_handle().reshape(self.shape)

    @staticmethod
    def buffer_init(vars_to_update):
        """
        Initialize global variables for multiprocessing workers.

        Args:
            vars_to_update (dict): Dictionary of variables to set globally
        """
        globals().update(vars_to_update)

    @classmethod
    def get(cls, name):
        """
        Get global variable by name.

        Args:
            name (str): Name of global variable

        Returns:
            Global variable value
        """
        return globals()[name]


class PearsonParallel:
    """
    Parallel computation of Pearson correlation matrices.

    This class provides efficient parallel computation of correlation matrices
    for large datasets using multiprocessing. It distributes the computation
    across multiple CPU cores for improved performance.

    The implementation uses a triangular computation approach to avoid
    redundant calculations and shared memory for efficient data exchange
    between processes.

    Basic usage:
        ```python
        import numpy as np
        from StatTools.auxiliary import PearsonParallel

        # Create large dataset
        data = np.random.normal(0, 1, (1000, 5000))

        # Compute correlation matrix in parallel
        corr_computer = PearsonParallel(data)
        correlation_matrix = corr_computer.create_matrix(threads=8, progress_bar=True)
        ```

    Args:
        input_array (ndarray): Input data matrix (n_variables x n_samples)

    Attributes:
        arr (ndarray): Input data array
        quantity (int): Number of variables
        length (int): Number of samples per variable

    Raises:
        NameError: If input array is 1D or has invalid dimensions

    Warning:
        For small datasets (< 10^5 elements), parallel processing may be slower
        due to overhead. Consider using numpy.corrcoef for small arrays.

    Note:
        The correlation matrix is symmetric with 1.0 on the diagonal.
        Only the upper triangle is computed to avoid redundancy.
    """

    def __init__(self, input_array):

        if isinstance(input_array, str):
            try:
                input_array = numpy.loadtxt(input_array)
            except OSError:
                error_str = (
                    "\n    The file either doesn't exit or you use a wrong path!"
                )
                raise NameError(error_str)

        if isinstance(input_array, list):
            try:
                input_array = numpy.array(input_array)
            except numpy.VisibleDeprecationWarning:
                error_str = (
                    "\n    Error occurred when converting list to numpy array! "
                    "\n    List probably has different dimensions!"
                )
                raise NameError(error_str)

        if input_array.ndim == 1:
            error_str = "\n    PearsonParallel got 1-dimensional array !"
            raise NameError(error_str)

        if numpy.size(input_array) < pow(10, 5):
            print(
                "\n    PearsonParallel Warning: Working in parallel mode with such small arrays is not effective !"
            )

        self.arr = input_array

        if len(input_array) > 2:
            self.triangle_result = self.triangle_divider(len(input_array))
            self.working_ranges = self.triangle_result[0]
            self.cells_to_count = self.triangle_result[1]
            self.quantity = self.arr.shape[0]
            self.length = self.arr.shape[1]

    def create_matrix(self, threads=cpu_count(), progress_bar=False):
        """
        Compute correlation matrix using parallel processing.

        Args:
            threads (int): Number of threads to use (default: CPU count)
            progress_bar (bool): Show progress bar if True

        Returns:
            ndarray: Correlation matrix of shape (n_variables, n_variables)

        Raises:
            NameError: If threads < 1
        """
        if threads < 1:
            error_str = (
                "\n    PearsonParallel Error: There is no point of calling this method using less than 2 "
                "threads since for loop is going to be faster!"
            )
            raise NameError(error_str)

        if len(self.arr) == 2:
            return numpy.corrcoef(self.arr[0], self.arr[1])[0][1]

        shared_array = Array(c_double, self.quantity * self.length, lock=True)
        numpy.copyto(
            numpy.frombuffer(shared_array.get_obj()).reshape(self.arr.shape), self.arr
        )
        del self.arr

        result_matrix = Array(c_double, self.quantity * self.quantity, lock=True)

        bar_counter = Value("i", 0)
        bar_lock = Lock()

        with closing(
            Pool(
                processes=threads,
                initializer=self.global_initializer,
                initargs=(shared_array, bar_counter, bar_lock, result_matrix),
            )
        ) as pool:
            pool.map(
                partial(self.corr_matrix, quantity=self.quantity, length=self.length),
                self.working_ranges,
            )

        ans = numpy.frombuffer((result_matrix.get_obj())).reshape(
            (self.quantity, self.quantity)
        )
        for i, j in zip(range(len(ans)), range(len(ans))):
            ans[i][j] = 1.0
        return ans

    @staticmethod
    def triangle_divider(field_size):
        """
        Divide correlation matrix computation into triangular regions.

        This method divides the correlation matrix computation into
        triangular regions that can be processed independently.

        Args:
            field_size (int): Size of the correlation matrix

        Returns:
            tuple: (working_ranges, cells_to_count)
        """
        cpu_available = cpu_count()
        cells_to_count = (field_size * field_size - field_size) / 2

        cycle = int(cells_to_count / cpu_available)

        cells = []
        start = [0, 0]
        on_interval = 0

        for r1 in range(field_size):
            for r2 in range(field_size):

                if r1 >= r2:
                    continue
                else:

                    if on_interval <= cycle:
                        on_interval += 1
                    else:
                        cells.append([start, [r1, r2]])
                        start = [r1, r2]
                        on_interval = 0

        cells.append([cells[-1][1], [field_size - 1, field_size - 1]])

        return [cells, cells_to_count]

    @staticmethod
    def global_initializer(arr, bar_val, bar_lock, result_arr):
        """
        Initialize global variables for parallel correlation computation.

        Args:
            arr: Shared input data array
            bar_val: Progress bar counter
            bar_lock: Progress bar lock
            result_arr: Shared result matrix
        """
        global shared_array
        global counter
        global lock
        global matrix

        shared_array = arr
        counter = bar_val
        lock = bar_lock
        matrix = result_arr

    @staticmethod
    def corr_matrix(working_range, quantity, length):
        """
        Compute correlations for a subset of matrix elements.

        Args:
            working_range: Range of matrix elements to compute
            quantity: Number of variables
            length: Number of samples per variable
        """

        def get_row(index):
            return numpy.frombuffer(shared_array.get_obj()).reshape((quantity, length))[
                index
            ]

        def write_to_matrix(value, r1, r2):
            numpy.frombuffer(matrix.get_obj()).reshape((quantity, quantity))[r1][
                r2
            ] = value

        start = working_range[0]
        stop = working_range[1]

        iterations_buffer = 0

        for r1 in range(start[0], stop[0] + 1):

            if r1 == stop[0]:
                r2_stop = stop[1]
            else:
                r2_stop = quantity

            if r1 == start[0]:
                r2_start = start[1]
            else:
                r2_start = r1 + 1

            for r2 in range(r2_start, r2_stop):

                if r1 > r2:
                    continue
                else:
                    corr_value = numpy.corrcoef(get_row(r1), get_row(r2))[0][1]
                    write_to_matrix(corr_value, r1, r2)
                    write_to_matrix(corr_value, r2, r1)

                    iterations_buffer += 1

                    if iterations_buffer >= 250:
                        with lock:
                            counter.value += iterations_buffer
                            iterations_buffer = 0

        with lock:
            counter.value += iterations_buffer


class CheckNumpy:
    """
    Descriptor for numpy array type checking and conversion.

    This descriptor automatically converts list inputs to numpy arrays
    and validates that assigned values are numpy arrays.

    Usage:
        ```python
        class MyClass:
            data = CheckNumpy()

            def __init__(self, data):
                self.data = data  # Automatically converted to numpy array
        ```

    Raises:
        ValueError: If assigned value cannot be converted to numpy array
    """

    def __set_name__(self, owner, name):
        """Set the attribute name for this descriptor."""
        self.name = name

    def __get__(self, instance, owner):
        """Get the attribute value."""
        return instance.__dict__[self.name]

    def __set__(self, instance, value):
        """Set the attribute value with type checking and conversion."""
        if isinstance(value, ndarray):
            instance.__dict__[self.name] = value
        elif isinstance(value, list):
            try:
                instance.__dict__[self.name] = array(value)
            except Exception:
                raise ValueError("Cannot cast input list to numpy array!")
        else:
            raise ValueError("Only list or numpy.ndarray can be used as input data!")
