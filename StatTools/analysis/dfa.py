import time
import warnings
from contextlib import closing
from ctypes import c_double
from functools import partial
from math import exp, floor
from multiprocessing import Array, Lock, Pool, Value, cpu_count
from threading import Thread
from typing import Tuple, Union

import numpy as np
from tqdm import TqdmWarning, tqdm

_CUPY_WARNED = False

# ====================== DFA core worker ======================


def _detrend_segment(
    y_segment: np.ndarray,
    indices: np.ndarray,
    degree: int,
) -> np.ndarray:
    """
    Compute detrended residuals for a single segment.

    Args:
        y_segment: Integrated time series segment (numpy.ndarray).
        indices: Indices for polynomial fitting (numpy.ndarray).
        degree: Polynomial degree for detrending.

    Returns:
        Residuals (detrended segment) as numpy.ndarray.
    """
    coef = np.polyfit(indices, y_segment, deg=degree)
    trend = np.polyval(coef, indices)
    residuals = y_segment - trend
    return residuals


def dfa_worker(
    indices: Union[int, list, np.ndarray],
    arr: Union[np.ndarray, None] = None,
    degree: int = 2,
    s_values: Union[list, np.ndarray, None] = None,
    n_integral: int = 1,
    backend: str = "cpu",
) -> list:
    """
    Core of the DFA algorithm. Processes a subset of series (indices) and
    returns fluctuation functions F^2(s).

    Args:
        indices: Indices of time series in the dataset to process.
        arr: Dataset array (must be 2D, shape: (n_series, length)).
        degree: Polynomial degree for detrending.
        s_values: Pre-calculated box sizes (scales).
        n_integral: Number of cumulative sum operations to apply (default: 1).
        backend: Computational backend ("cpu" or "gpu").

    Returns:
        list of (s, F2_s) for each requested index, where F2_s is F^2(s).
    """

    data = np.asarray(arr, dtype=float)

    if data.ndim != 2:
        raise ValueError(
            f"dfa_worker expects 2D array, got {data.ndim}D array. "
            f"Normalize data to 2D before calling (use reshape(1, -1) for 1D)."
        )

    indices = np.atleast_1d(indices).astype(int).ravel()

    n_series = data.shape[0]
    for idx in indices:
        if not (0 <= idx < n_series):
            raise IndexError(
                f"Index {idx} out of bounds for array with {n_series} series"
            )

    series_len_global = data.shape[1]

    if s_values is None:
        s_max = int(series_len_global / 4)
        s_values = [int(exp(step)) for step in np.arange(1.6, np.log(s_max), 0.5)]
    else:
        if isinstance(s_values, np.ndarray):
            s_values = s_values.tolist()
        s_values = list(s_values)

    results = []

    if backend not in ("cpu", "gpu"):
        raise ValueError(f'backend must be "cpu" or "gpu", got: {backend!r}')

    cp = None
    if backend == "gpu":
        try:
            import cupy as cp
        except ImportError:
            global _CUPY_WARNED
            if not _CUPY_WARNED:
                warnings.warn(
                    "CuPy not available, GPU acceleration disabled. "
                    "Switching to CPU backend. "
                    "Install with: pip install cupy-cuda11x (or cupy-cuda12x)",
                    UserWarning,
                    stacklevel=2,
                )
                _CUPY_WARNED = True
            backend = "cpu"

    if backend == "gpu":
        batch_data = cp.asarray(data[indices], dtype=cp.float64)
        n_series_batch = batch_data.shape[0]
        series_len = batch_data.shape[1]

        y_batch = batch_data - cp.mean(batch_data, axis=1, keepdims=True)
        for _ in range(n_integral):
            y_batch = cp.cumsum(y_batch, axis=1)

        s_list_all = []
        f2_gpu_list = []  # Store GPU arrays

        for s_val in s_values:
            if s_val >= series_len / 4:
                continue

            cycles_amount = floor(series_len / s_val)

            if cycles_amount < 1:
                continue

            n_segments = cycles_amount - 1

            if n_segments == 0:
                continue

            cutoff = 1 + n_segments * s_val
            y_sliced = y_batch[:, 1:cutoff]

            # Reshape into flat segments
            flat_segments = y_sliced.reshape(n_series_batch * n_segments, s_val)

            x = cp.arange(1, s_val + 1, dtype=cp.float64)
            x_centered = x - 0.5 * s_val

            # Polynomial fitting using normal equations
            V = cp.vander(x_centered, degree + 1, increasing=False)
            VTV_inv = cp.linalg.inv(V.T @ V)
            proj_matrix = V @ VTV_inv

            # Compute coefficients for all segments at once
            coeffs = flat_segments @ proj_matrix

            trends = coeffs @ V.T

            # Compute F^2(s): mean squared residuals per segment, then mean across segments per series
            f2_per_seg = cp.mean((flat_segments - trends) ** 2, axis=1)
            f2_batch = cp.mean(f2_per_seg.reshape(n_series_batch, n_segments), axis=1)

            s_list_all.append(s_val)
            f2_gpu_list.append(f2_batch)

        # Transfer all results from GPU to CPU
        if not f2_gpu_list:
            return [(np.array([]), np.array([])) for _ in range(n_series_batch)]

        all_f2_cpu = cp.stack(f2_gpu_list).get()

        for series_idx in range(n_series_batch):
            s_array = np.array(s_list_all)
            f2_array = all_f2_cpu[:, series_idx]
            results.append((s_array, f2_array))

    if backend == "cpu":
        for idx in indices:
            series = data[idx]

            if series.ndim > 1:
                series = series.flatten()

            data_centered = series - np.mean(series)
            y_cumsum = data_centered
            for _ in range(n_integral):
                y_cumsum = np.cumsum(y_cumsum)

            series_len = len(data_centered)

            s_list = []
            f2_list = []

            for s_val in s_values:
                if s_val >= series_len / 4:
                    continue

                s = np.arange(1, s_val + 1, dtype=int)
                cycles_amount = floor(series_len / s_val)

                if cycles_amount < 1:
                    continue

                f2_sum = 0.0
                s_temp = s.copy()

                for i in range(1, cycles_amount):
                    # Center indices for polynomial fitting: subtract (i + 0.5) * s_val
                    indices_s = (s_temp - (i + 0.5) * s_val).astype(int)
                    y_cumsum_s = y_cumsum[s_temp]

                    residuals = _detrend_segment(y_cumsum_s, indices_s, degree)
                    f2 = np.sum(residuals**2) / s_val

                    f2_sum += f2
                    s_temp += s_val

                f2_s = f2_sum / (cycles_amount - 1)
                s_list.append(s_val)
                f2_list.append(f2_s)

            s_array = np.array(s_list)
            f2_array = np.array(f2_list)
            results.append((s_array, f2_array))

    return results


# ====================== High-level DFA function ======================


def dfa(
    dataset,
    degree: int = 2,
    processes: int = 1,
    n_integral: int = 1,
    backend: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implementation of the Detrended Fluctuation Analysis (DFA) method.

    The algorithm removes local polynomial trends in integrated time series and
    returns the fluctuation function F^2(s) for each series.

    Args:
        dataset (ndarray): 1D or 2D array of time series data.
        degree (int): Polynomial degree for detrending (default: 2).
        processes (int): Number of parallel workers (default: 1).
            Note: If backend="gpu", multiprocessing is disabled and GPU is used instead.
        n_integral (int): Number of cumulative sum operations to apply (default: 1).
        backend (str): Computational backend. Options: "cpu" (default), "gpu".
            GPU mode uses single process (multiprocessing disabled).

    Returns:
        tuple: (s, F2_s)
            - For 1D input: two 1D arrays s, F2_s.
            - For 2D input:
                s is a 1D array (same scales for all series),
                F2_s is a 2D array where each row is F^2(s) for one time series.
    """
    data = np.asarray(dataset, dtype=float)
    if data.size == 0:
        raise ValueError("Input dataset is empty.")

    if data.ndim == 1:
        data = data.reshape(1, -1)
        single_series = True
    elif data.ndim == 2:
        single_series = False
    else:
        raise ValueError("Only 1D or 2D arrays are allowed!")

    series_len = data.shape[1]
    s_max = int(series_len / 4)
    s_values = [int(exp(step)) for step in np.arange(1.6, np.log(s_max), 0.5)]

    n_series = data.shape[0]

    # Validate backend
    if backend not in ("cpu", "gpu"):
        raise ValueError(f'backend must be "cpu" or "gpu", got: {backend!r}')

    if backend == "gpu":
        try:
            import cupy as cp
        except ImportError:
            global _CUPY_WARNED
            if not _CUPY_WARNED:
                warnings.warn(
                    "CuPy not available, GPU acceleration disabled. "
                    "Switching to CPU backend. "
                    "Install with: pip install cupy-cuda11x (or cupy-cuda12x)",
                    UserWarning,
                    stacklevel=2,
                )
                _CUPY_WARNED = True
            backend = "cpu"

    if backend == "gpu" and processes > 1:
        warnings.warn(
            f"GPU acceleration enabled: multiprocessing (processes={processes}) "
            f"is disabled. Using single GPU process instead.",
            UserWarning,
        )

    if backend == "gpu" or processes <= 1:
        indices = np.arange(n_series)
        results = dfa_worker(
            indices=indices,
            arr=data,
            degree=degree,
            s_values=s_values,
            n_integral=n_integral,
            backend=backend,
        )
    else:
        processes = min(processes, cpu_count(), n_series)
        chunks = np.array_split(np.arange(n_series), processes)

        worker_func = partial(
            dfa_worker,
            arr=data,
            degree=degree,
            s_values=s_values,
            n_integral=n_integral,
            backend="cpu",
        )

        results_list_of_lists = []
        with closing(Pool(processes=processes)) as pool:
            for sub in pool.map(worker_func, chunks):
                results_list_of_lists.append(sub)

        flat_results = []
        for sub in results_list_of_lists:
            flat_results.extend(sub)
        results = flat_results

    s_list = [r[0] for r in results]
    f2_list = [r[1] for r in results]

    if single_series:
        s_out = s_list[0]
        f2_out = f2_list[0]
    else:
        s_out = s_list[0]
        f2_out = np.vstack(f2_list)

    return s_out, f2_out


# ====================== Progress bar manager ======================


def bar_manager(description, total, counter, lock, mode="total", stop_bit=None):
    """
    Manages progress bar display for long-running operations.

    Args:
        description (str): Description text for the progress bar.
        total (int): Total number of items to process.
        counter (Value): Shared counter for tracking progress.
        lock (Lock): Thread lock for safe counter access.
        mode (str): Display mode - "total" (absolute) or "percent".
        stop_bit (Value): Optional stop signal for early termination.

    Returns:
        None
    """
    max_val = total
    if mode == "percent":
        max_val = 100
    with closing(tqdm(desc=description, total=max_val, leave=False, position=0)) as bar:
        try:
            last_val = counter.value
            while True:
                if stop_bit is not None and stop_bit.value > 0:
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


# ====================== DFA wrapper class ======================


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

    Args:
        dataset (array-like): Input time series data. Can be:
            - 1D numpy array for single time series
            - 2D numpy array for multiple time series (shape: n_series x length)
            - Path to text file containing data
        degree (int): Polynomial degree for detrending (default: 2)
        root (bool): Kept for backward compatibility, not used in current implementation (default: False)
        ignore_input_control (bool): Skip input validation (default: False)

    Attributes:
        dataset (numpy.ndarray): Processed input data
        degree (int): Polynomial degree for detrending
        root (bool): Root fluctuation flag (kept for backward compatibility, not used)
        s (numpy.ndarray): Scale values used in analysis
        F_s (numpy.ndarray): Fluctuation function squared values F^2(s)

    Raises:
        NameError: If input file doesn't exist or has invalid format
        ValueError: If input array has unsupported dimensions
    """

    def __init__(
        self,
        dataset: Union[np.ndarray, list, str],
        degree: int = 2,
        root: bool = False,
        ignore_input_control: bool = False,
    ) -> None:
        """
        Initialize DFA analyzer.

        Args:
            dataset: Input time series data (1D or 2D array, or file path).
            degree: Polynomial degree for detrending (default: 2).
            root: Kept for backward compatibility, not used in current implementation.
            ignore_input_control: Skip input validation (default: False).
        """
        self.degree = degree
        self.root = root
        self.dataset = None
        self.s = None
        self.F_s = None

        if ignore_input_control:
            scales, fluct2_values = dfa(
                dataset,
                degree=self.degree,
                processes=1,
                n_integral=1,
            )
            self.s = scales
            self.F_s = fluct2_values
        else:
            if isinstance(dataset, str):
                try:
                    dataset = np.loadtxt(dataset)
                except OSError:
                    raise NameError(
                        "\n    The file either doesn't exist or the path is wrong!"
                    )
                if np.size(dataset) == 0:
                    raise NameError("\n    Input file is empty!")

            if not isinstance(dataset, np.ndarray):
                try:
                    dataset = np.array(dataset, dtype=float)
                except (ValueError, TypeError):
                    raise NameError(
                        "\n    Input dataset is supposed to be numpy array, list or filepath!"
                    )

            if dataset.ndim > 2 or dataset.ndim == 0:
                raise NameError("\n    Only 1- or 2-dimensional arrays are allowed!")

            series_len = len(dataset) if dataset.ndim == 1 else dataset.shape[1]
            if series_len < 20:
                raise NameError("Wrong input array ! (It's probably too short)")

            self.dataset = np.array(dataset)

    @staticmethod
    def _hurst_exponent(
        x_axis: np.ndarray, y_axis: np.ndarray, simple_mode: bool = True
    ) -> float:
        """
        Calculate the slope of linear regression between x_axis and y_axis.

        Args:
            x_axis: Independent variable array.
            y_axis: Dependent variable array.
            simple_mode: If True, use linear fit (default: True).

        Returns:
            float: Slope of the linear regression.
        """
        if not simple_mode:
            error_str = "\n    Non-linear approximation is not supported yet!"
            raise NameError(error_str)

        slope = np.polyfit(x_axis, y_axis, deg=1)[0]
        return slope

    @staticmethod
    def initializer_for_parallel_mod(
        shared_array: Array,
        h_est: Array,
        shared_c: Value,
        shared_l: Lock,
    ) -> None:
        """
        Initialize global variables for parallel processing.
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
    def dfa_core_cycle(
        dataset: Union[np.ndarray, list],
        degree: int,
        root: bool,
        backend: str = "cpu",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform DFA for a single vector and return (log(s), log(F(s))).

        For backward compatibility, returns the same format as the original implementation:
        - log(s): logarithm of scale values
        - log(F(s)): logarithm of fluctuation function values

        Args:
            dataset: Input time series data (1D or 2D).
            degree: Polynomial degree for detrending.
            root: Kept for backward compatibility, not used in current implementation.
            backend: Computational backend ("cpu" or "gpu").
                    Default is "cpu".
        """
        data = np.asarray(dataset, dtype=float)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        result_list = dfa_worker(
            indices=0,
            arr=data,
            degree=degree,
            s_values=None,
            n_integral=1,
            backend=backend,
        )
        scales, fluct2_values = result_list[0]

        # Convert to log format for backward compatibility
        log_s = np.log(scales)
        # Extract F(s) from F^2(s) and take logarithm
        log_f = np.log(np.sqrt(fluct2_values))

        return log_s, log_f

    def find_h(
        self, simple_mode: bool = True, backend: str = "cpu"
    ) -> Union[float, np.ndarray]:
        """
        Estimate the Hurst exponent from fluctuation analysis.

        External interface is preserved:
        - returns scalar for 1D input;
        - returns 1D array for 2D input.

        Args:
            simple_mode: Kept for backward compatibility, not used
                        (always uses linear fit).
            backend: Computational backend ("cpu" or "gpu").
                    Default is "cpu".
        """
        if not simple_mode:
            error_str = "\n    Non-linear approximation is not supported yet!"
            raise NameError(error_str)

        if self.dataset is None:
            raise NameError("\n    Dataset is not initialized for DFA.find_h!")

        scales, fluct2_values = dfa(
            self.dataset,
            self.degree,
            processes=1,
            n_integral=1,
            backend=backend,
        )
        self.s, self.F_s = scales, fluct2_values

        log_s = np.log(self.s)

        if self.dataset.ndim == 1:
            log_f = np.log(np.sqrt(self.F_s))
            return self._hurst_exponent(log_s, log_f, simple_mode=True)
        else:
            n_series = self.dataset.shape[0]
            h_array = np.empty(n_series, dtype=float)
            for i in range(n_series):
                log_f = np.log(np.sqrt(self.F_s[i]))
                h_array[i] = self._hurst_exponent(log_s, log_f, simple_mode=True)
            return h_array

    def parallel_2d(
        self,
        threads: int = cpu_count(),
        progress_bar: bool = False,
        h_control: bool = False,
        h_target: float = 0.0,
        h_limit: float = 0.0,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Parallel computation of Hurst exponents for 2D datasets.

        Args:
            threads (int): Number of parallel processes (default: CPU count).
            progress_bar (bool): Show progress bar if True (default: False).
            h_control (bool): Enable Hurst exponent control mode (default: False).
            h_target (float): Target Hurst exponent for control mode.
            h_limit (float): Acceptable deviation from target H.

        Returns:
            numpy.ndarray or tuple: Hurst exponents, or (H_values, invalid_indices) if h_control is True.
        """
        result = None

        if threads == 1 or self.dataset.ndim == 1:
            result = self.find_h()
        elif len(self.dataset) / threads < 1:
            warnings.warn(
                f"Input array is too small for using it in parallel mode! "
                f"You should either use fewer threads ({len(self.dataset)}) or avoid parallel mode!",
                UserWarning,
            )
            result = self.find_h()
        else:
            vectors_indices_by_threads = np.array_split(
                np.linspace(0, len(self.dataset) - 1, len(self.dataset), dtype=int),
                threads,
            )

            dataset_to_memory = Array(
                c_double, len(self.dataset) * len(self.dataset[0])
            )
            h_estimation_in_memory = Array(c_double, len(self.dataset))

            np.copyto(
                np.frombuffer(dataset_to_memory.get_obj()).reshape(
                    (len(self.dataset), len(self.dataset[0]))
                ),
                self.dataset,
            )

            shared_counter = Value("i", 0)
            shared_lock = Lock()

            if progress_bar:
                bar_thread = Thread(
                    target=bar_manager,
                    args=("DFA", len(self.dataset), shared_counter, shared_lock),
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

            if progress_bar:
                bar_thread.join()

            if h_control:
                invalid_i = np.concatenate(invalid_i)
                result = (np.frombuffer(h_estimation_in_memory.get_obj()), invalid_i)
            else:
                result = np.frombuffer(h_estimation_in_memory.get_obj())

        return result

    def parallel_core(
        self,
        indices: np.ndarray,
        quantity: int,
        length: int,
        h_control: bool,
        h_target: float,
        h_limit: float,
    ) -> np.ndarray:
        """
        Core function for parallel computation of Hurst exponents.

        Args:
            indices: Array indices to process.
            quantity: Total number of time series.
            length: Length of each time series.
            h_control: Enable Hurst exponent control.
            h_target: Target Hurst exponent.
            h_limit: Acceptable deviation limit.

        Returns:
            numpy.ndarray: Array of invalid indices if h_control enabled,
                         empty array otherwise.
        """
        invalid_i = []

        all_data = np.frombuffer(datasets_array.get_obj()).reshape((quantity, length))

        for i in indices:
            vector = all_data[i]

            vector_2d = vector.reshape(1, -1)

            result_list = dfa_worker(
                indices=0,
                arr=vector_2d,
                degree=self.degree,
                s_values=None,
                n_integral=1,
                backend="cpu",
            )
            scales, fluct2_values = result_list[0]

            log_s = np.log(scales)
            log_f = np.log(np.sqrt(fluct2_values))
            h_calc = self._hurst_exponent(log_s, log_f, simple_mode=True)

            np.frombuffer(estimations.get_obj())[i] = h_calc

            with shared_lock:
                shared_counter.value += 1

            if h_control and abs(h_calc - h_target) > h_limit:
                invalid_i.append(i)

        return np.array(invalid_i)
