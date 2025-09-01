import time
from contextlib import closing
from ctypes import c_double
from functools import partial
from math import ceil, exp, floor
from multiprocessing import Array, Lock, Pool, Value, cpu_count
from threading import Thread

import numpy
from tqdm import TqdmWarning, tqdm


def bar_manager(description, total, counter, lock, mode="total", stop_bit=None):
    """
    Manages progress bar display for long-running operations.

    Args:
        description (str): Description text for the progress bar
        total (int): Total number of items to process
        counter (Value): Shared counter for tracking progress
        lock (Lock): Thread lock for safe counter access
        mode (str): Display mode - "total" or "percent"
        stop_bit (Value): Optional stop signal for early termination

    Returns:
        None: Displays progress bar until completion
    """
    max_val = total
    if mode == "percent":
        max_val = 100
    with closing(tqdm(desc=description, total=max_val, leave=False, position=0)) as bar:

        try:
            last_val = counter.value
            while True:
                if stop_bit is not None:
                    if stop_bit.value > 0:
                        break

                time.sleep(0.25)
                with lock:
                    if counter.value > last_val:
                        if mode == "percent":
                            bar.update(
                                round(((counter.value - last_val) * 100 / total), 2)
                            )
                        else:
                            bar.update(counter.value - last_val)
                        last_val = counter.value

                    if counter.value == total:
                        bar.close()
                        break
        except TqdmWarning:
            return None


class DFA:
    """
    Detrended Fluctuation Analysis (DFA) implementation for estimating Hurst exponent.

    DFA is a method for determining the statistical self-affinity of a signal by analyzing
    its fluctuation function F(s) as a function of time scale s. The Hurst exponent H
    is estimated from the scaling relationship F(s) ~ s^H.

    For fractional Brownian motion with Hurst exponent H:
    - H < 0.5: Anti-persistent behavior
    - H = 0.5: Random walk (uncorrelated)
    - H > 0.5: Persistent behavior (long-range correlations)

    Basic usage:
        ```python
        import numpy as np
        from StatTools.analysis.dfa import DFA

        # Generate sample data with known Hurst exponent
        data = np.random.normal(0, 1, 10000)

        # Create DFA analyzer
        dfa = DFA(data, degree=2, root=False)

        # Estimate Hurst exponent
        hurst_exponent = dfa.find_h()

        # For 2D data (multiple time series)
        data_2d = np.random.normal(0, 1, (10, 10000))
        dfa_2d = DFA(data_2d)
        hurst_exponents = dfa_2d.find_h()  # Returns array of H values
        ```

    Args:
        dataset (array-like): Input time series data. Can be:
            - 1D numpy array for single time series
            - 2D numpy array for multiple time series (shape: n_series x length)
            - Path to text file containing data
        degree (int): Polynomial degree for detrending (default: 2)
        root (bool): If True, use root-mean-square fluctuations F(s) ~ s^{H-1}
                    If False, use standard fluctuations F(s) ~ s^H (default: False)
        ignore_input_control (bool): Skip input validation (default: False)

    Attributes:
        dataset (numpy.ndarray): Processed input data
        degree (int): Polynomial degree for detrending
        root (bool): Root fluctuation flag
        s (numpy.ndarray): Scale values used in analysis
        F_s (numpy.ndarray): Fluctuation function values

    Raises:
        NameError: If input file doesn't exist or has invalid format
        ValueError: If input array has unsupported dimensions
    """

    def __init__(self, dataset, degree=2, root=False, ignore_input_control=False):
        if ignore_input_control:
            s_return_1d, F_s_return_1d = self.dfa_core_cycle(dataset, degree, root)
            self.s = s_return_1d
            self.F_s = F_s_return_1d
        else:
            if isinstance(dataset, type("string")):
                try:
                    dataset = numpy.loadtxt(dataset)
                except OSError:
                    error_str = (
                        "\n    The file either doesn't exit or you use wrong path!"
                    )
                    raise NameError(error_str)

                if numpy.size(dataset) == 0:
                    error_str = "\n    Input file is empty!"
                    raise NameError(error_str)

            if not isinstance(dataset, type(numpy.array([]))):
                try:  # in case of list
                    dataset = numpy.array(dataset, dtype=float)
                except ValueError:
                    error_str = "\n    Input dataset is supposed to be numpy array, list or directory!"
                    raise NameError(error_str)

            dataset = numpy.array(dataset)

            if dataset.ndim > 2 or dataset.ndim == 0:
                error_str = "\n    You can not use such input array! Only 1- or 2-dimensional arrays are allowed!"
                raise NameError(error_str)

            self.dataset = dataset
            self.degree = degree
            self.root = root

            if self.dataset.ndim == 1:
                s_max = int(len(dataset) / 4)
                try:
                    log_s_max = numpy.arange(1.6, numpy.log(s_max), 0.5)
                except ValueError:
                    error_str = "\n    Wrong input array ! (It's probably too short)"
                    raise NameError(error_str)
                if numpy.size(log_s_max) < 1:
                    error_str = "\n    Input array is too small! (It usually requires 20 or more samples!)"
                    raise NameError(error_str)

            if self.dataset.ndim == 2:

                s_max = int(len(dataset[0]) / 4)
                try:
                    log_s_max = numpy.arange(1.6, numpy.log(s_max), 0.5)
                except ValueError:
                    error_str = "\n    Wrong input vectors in input matrix! (They are probably too short)"
                    raise NameError(error_str)
                if numpy.size(log_s_max) < 1:
                    error_str = (
                        "\n   Vectors in your input array are too short! Use longer vectors "
                        "(it usually requires 20 or more samples) or transpose!"
                    )
                    raise NameError(error_str)

    @staticmethod
    def initializer_for_parallel_mod(shared_array, h_est, shared_c, shared_l):
        """
        Initialize global variables for parallel processing.

        Args:
            shared_array: Shared memory array containing datasets
            h_est: Shared array for storing Hurst exponent estimates
            shared_c: Shared counter for progress tracking
            shared_l: Shared lock for thread safety
        """
        global datasets_array
        global estimations
        global shared_counter
        global shared_lock
        datasets_array = shared_array
        estimations = h_est
        shared_counter = shared_c
        shared_lock = shared_l

    @staticmethod
    def dfa_core_cycle(dataset, degree, root):
        """
        Core DFA algorithm implementation.

        Computes the fluctuation function F(s) for different time scales s by:
        1. Integrating the time series to get cumulative sum Y(i)
        2. Dividing Y(i) into segments of length s
        3. Fitting polynomial of degree 'degree' to each segment
        4. Computing RMS fluctuation F(s) of detrended segments
        5. Repeating for different scales s

        Args:
            dataset (numpy.ndarray): Input time series
            degree (int): Polynomial degree for detrending
            root (bool): Use root fluctuations if True

        Returns:
            tuple: (s_values, F_s_values) where:
                - s_values: Array of scale values (log-spaced)
                - F_s_values: Corresponding fluctuation function values
        """
        data_mean = numpy.mean(dataset)
        data = dataset - data_mean
        Y_cumsum = numpy.cumsum(data)

        s_max = int(len(data) / 4)

        log_s_max = numpy.arange(1.6, numpy.log(s_max), 0.5)

        x_Axis = []
        y_Axis = []

        for step in log_s_max:
            s = numpy.linspace(1, floor(exp(step)), floor(exp(step)), dtype=int)
            cycles_amount = floor(len(data) / len(s))

            F_q_s_sum = 0
            for i in range(1, cycles_amount):
                indices = numpy.array((s - (i + 0.5) * len(s)), dtype=int)
                Y_cumsum_s = numpy.take(Y_cumsum, s)

                coef = numpy.polyfit(indices, Y_cumsum_s, deg=degree)
                current_trend = numpy.polyval(coef, indices)
                F_2 = sum(pow((Y_cumsum_s - current_trend), 2)) / len(s)
                F_q_s_sum += pow(F_2, (degree / 2))
                s += floor(exp(step))

            F1 = pow(((1 / cycles_amount) * F_q_s_sum), 1 / degree)
            x_Axis.append(numpy.log(floor(exp(step))))
            if root:
                y_Axis.append(numpy.log(F1 / numpy.sqrt(len(s))))
            else:
                y_Axis.append(numpy.log(F1))

        return numpy.array(x_Axis), numpy.array(y_Axis)

    def find_h(self, simple_mode=True):
        """
        Estimate Hurst exponent from fluctuation analysis.

        Performs DFA on the dataset and fits a linear regression to the
        log-log plot of F(s) vs s to estimate the Hurst exponent.

        Args:
            simple_mode (bool): If True, use simple linear regression.
                               If False, non-linear fitting (not implemented)

        Returns:
            float or numpy.ndarray: Hurst exponent(s). Returns:
                - Single float for 1D input data
                - Array of floats for 2D input data (one per time series)

        Raises:
            NameError: If non-linear mode is requested (not implemented)
        """
        if self.dataset.ndim == 1:
            self.s, self.F_s = self.dfa_core_cycle(self.dataset, self.degree, self.root)
        else:
            self.s = numpy.array([])
            self.F_s = numpy.array([])
            for vector in self.dataset:
                s, F_s = self.dfa_core_cycle(vector, self.degree, self.root)
                if numpy.size(self.s) < 1:
                    self.s = s
                    self.F_s = F_s
                else:
                    self.s = numpy.vstack((self.s, s))
                    self.F_s = numpy.vstack((self.F_s, F_s))

        if simple_mode:

            if self.s.ndim == 1:
                return numpy.polyfit(self.s, self.F_s, deg=1)[0]
            else:
                h_estimation = []
                for s, F_s in zip(self.s, self.F_s):
                    h_estimation.append(numpy.polyfit(s, F_s, deg=1)[0])
                return numpy.array(h_estimation)
        else:
            error_str = "\n    Non-linear approximation is non supported yet!"
            raise NameError(error_str)

    def parallel_2d(
        self,
        threads=cpu_count(),
        progress_bar=False,
        h_control=False,
        h_target=float(),
        h_limit=float(),
    ):
        """
        Perform parallel DFA analysis on 2D datasets.

        Processes multiple time series in parallel using multiprocessing.
        Useful for large datasets with many time series.

        Args:
            threads (int): Number of parallel processes (default: CPU count)
            progress_bar (bool): Show progress bar if True
            h_control (bool): Enable Hurst exponent control mode
            h_target (float): Target Hurst exponent for control mode
            h_limit (float): Acceptable deviation from target H

        Returns:
            numpy.ndarray or tuple: Hurst exponents, or (H_values, invalid_indices)
                                 if h_control is True

        Raises:
            ValueError: If dataset is too small for parallel processing
        """
        if threads == 1 or self.dataset.ndim == 1:
            return self.find_h()

        if len(self.dataset) / threads < 1:
            error_str = (
                "\n    DFA Warning: Input array is too small for using it in parallel mode!"
                f"\n    You better use either less threads ({len(self.dataset)}) or don't use "
                f"parallel mode at all!"
            )
            print(error_str)
            h_est = self.find_h()
            return h_est

        if len(self.dataset) / threads < 10:
            error_str = (
                "\n    DFA Warning: It may be not  so effective when using parallel mode with such small array!"
                "\n    Spawning processes creates its own overhead!"
            )
            print(error_str)

        vectors_indices_by_threads = numpy.array_split(
            numpy.linspace(0, len(self.dataset) - 1, len(self.dataset), dtype=int),
            threads,
        )

        dataset_to_memory = Array(c_double, len(self.dataset) * len(self.dataset[0]))
        h_estimation_in_memory = Array(c_double, len(self.dataset))
        numpy.copyto(
            numpy.frombuffer(dataset_to_memory.get_obj()).reshape(
                (len(self.dataset), len(self.dataset[0]))
            ),
            self.dataset,
        )

        shared_counter = Value("i", 0)
        shared_lock = Lock()

        if progress_bar:
            bar_thread = Thread(
                target=bar_manager,
                args=(f"DFA", len(self.dataset), shared_counter, shared_lock),
            )
            bar_thread.start()

        with closing(
            Pool(
                processes=threads,
                initializer=self.initializer_for_parallel_mod,
                initargs=(
                    dataset_to_memory,
                    h_estimation_in_memory,
                    shared_counter,
                    shared_lock,
                ),
            )
        ) as pool:
            invalid_i = pool.map(
                partial(
                    self.parallel_core,
                    quantity=len(self.dataset),
                    length=len(self.dataset[0]),
                    h_control=h_control,
                    h_target=h_target,
                    h_limit=h_limit,
                ),
                vectors_indices_by_threads,
            )

        if h_control:
            invalid_i = numpy.concatenate(invalid_i)
            return numpy.frombuffer(h_estimation_in_memory.get_obj()), invalid_i
        else:
            return numpy.frombuffer(h_estimation_in_memory.get_obj())

    def parallel_core(self, indices, quantity, length, h_control, h_target, h_limit):
        """
        Core parallel processing function for DFA analysis.

        Processes a subset of time series indices in parallel.

        Args:
            indices (numpy.ndarray): Array indices to process
            quantity (int): Total number of time series
            length (int): Length of each time series
            h_control (bool): Enable Hurst exponent control
            h_target (float): Target Hurst exponent
            h_limit (float): Acceptable deviation limit

        Returns:
            numpy.ndarray: Array of invalid indices if h_control enabled,
                          empty array otherwise
        """
        invalid_i = []
        for i in indices:
            vector = numpy.frombuffer(datasets_array.get_obj()).reshape(
                (quantity, length)
            )[i]
            x_ax, y_ax = self.dfa_core_cycle(vector, self.degree, self.root)
            lin_reg = numpy.polyfit(x_ax, y_ax, deg=1)[0]
            numpy.frombuffer(estimations.get_obj())[i] = lin_reg
            with shared_lock:
                shared_counter.value += 1

            if h_control:
                if abs(lin_reg - h_target) > h_limit:
                    invalid_i.append(i)

        return numpy.array(invalid_i)
