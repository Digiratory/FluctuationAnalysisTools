import time
import warnings
from collections.abc import Sequence
from contextlib import closing
from ctypes import c_double
from functools import partial
from math import exp, floor
from multiprocessing import Array, Lock, Pool, Value, cpu_count
from threading import Thread
from typing import Tuple, Union

import numpy as np
from tqdm import TqdmWarning, tqdm

# ====================== DFA core worker ======================


def _detrend_segment(
    y_segment: np.ndarray, indices: np.ndarray, degree: int, xp=np
) -> np.ndarray:
    """
    Compute detrended residuals for one segment or a matrix of segments.

    Args:
        y_segment: Integrated segment or 2D array with segments in columns.
        indices: Indices for polynomial fitting.
        degree: Polynomial degree for detrending.
        xp: NumPy or CuPy, matching the input arrays (default: NumPy).

    Returns:
        Residuals with the same shape as y_segment.
    """
    coef = xp.polyfit(indices, y_segment, deg=degree)
    # This polyval accepts several polynomials and ascending coefficients.
    trend = xp.polynomial.polynomial.polyval(indices, coef[::-1]).T
    residuals = y_segment - trend
    return residuals


def dfa_worker(
    indices: Union[int, list, np.ndarray],
    arr: Union[np.ndarray, None] = None,
    degree: int = 2,
    s_values: Union[list, np.ndarray, None] = None,
    n_integral: int = 1,
    xp=np,
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
        xp: Array library used for calculations: NumPy or CuPy (default: NumPy).

    Returns:
        list of (s, F2_s) for each requested index, where F2_s is F^2(s).
        Scales are NumPy arrays; F2_s arrays belong to the selected library.
    """
    data = xp.asarray(arr, dtype=float)

    if data.ndim != 2:
        raise ValueError(
            f"dfa_worker expects 2D array, got {data.ndim}D array. "
            f"Normalize data to 2D before calling (use reshape(1, -1) for 1D)."
        )

    if not isinstance(indices, (list, np.ndarray)):
        indices = [indices]

    series_len_global = data.shape[1]
    if s_values is None:
        s_max = int(series_len_global / 4)
        s_values = [int(exp(step)) for step in np.arange(1.6, np.log(s_max), 0.5)]
    else:
        s_values = list(s_values)

    results = []

    for idx in indices:
        series = data[idx]

        # Standard DFA preprocessing: mean-centering and integration
        data_centered = series - xp.mean(series)
        y_cumsum = data_centered
        for _ in range(n_integral):
            y_cumsum = xp.cumsum(y_cumsum)
        series_len = len(data_centered)

        s_list = []
        f2_list = []

        for s_val in s_values:
            if s_val >= series_len / 4:
                continue

            s = xp.arange(1, s_val + 1, dtype=int)
            len_s = len(s)
            cycles_amount = floor(series_len / len_s)
            if cycles_amount < 1:
                continue

            # Keep the original windows: start at index 1, use cycles_amount - 1.
            n_segments = cycles_amount - 1
            segments = y_cumsum[1 : 1 + n_segments * len_s].reshape(n_segments, len_s).T
            indices_s = (s - 1.5 * len_s).astype(int)

            # Fit all windows of this scale in one library call.
            residuals = _detrend_segment(segments, indices_s, degree, xp=xp)
            f2 = xp.sum(residuals**2, axis=0) / len_s

            # Preserve the original normalization by cycles_amount.
            f2_s = xp.sum(f2) / cycles_amount
            s_list.append(s_val)
            f2_list.append(f2_s)

        # Stack on the selected device: CuPy scalars cannot be converted
        # implicitly into a NumPy array.
        f2_values = xp.stack(f2_list) if f2_list else xp.empty(0, dtype=float)
        results.append((np.array(s_list), f2_values))

    return results


# ====================== High-level DFA function ======================


def dfa(
    dataset,
    degree: int = 2,
    processes: int = 1,
    n_integral: int = 1,
    s_values: Union[int, Sequence, None] = None,
    s_min: int = 5,
    backend: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implementation of the Detrended Fluctuation Analysis (DFA) method.

    The algorithm removes local polynomial trends in integrated time series and
    returns the fluctuation function F^2(s) for each series.

    Args:
        dataset (ndarray): 1D or 2D array of time series data.
        degree (int): Polynomial degree for detrending (default: 2).
        processes (int): Number of CPU workers (default: 1). Ignored with a
            warning when CUDA is used and processes > 1.
        n_integral (int): Number of cumulative sum operations to apply (default: 1).
        s_values (Union[int, Sequence, None]): points where  fluctuation function F^2(s) is calculated (default: None).
        s_min (int): Minimal scale in s_values (default: 5).
        backend (str): "cpu" uses NumPy; "cuda" uses CuPy.
            Falls back to CPU with a warning if CuPy cannot be imported.

    Returns:
        tuple: (s, F2_s), both NumPy arrays, including for the CUDA backend.
            - For 1D input: two 1D arrays s, F2_s.
            - For 2D input:
                s is a 1D array (same scales for all series),
                F2_s is a 2D array where each row is F^2(s) for one time series.
    """
    if backend not in ("cpu", "cuda"):
        raise ValueError("backend must be 'cpu' or 'cuda'")

    xp = np
    if backend == "cuda":
        try:
            import cupy as xp
        except ImportError:
            warnings.warn(
                "DFA warning: CuPy could not be imported; "
                "falling back to the CPU backend. "
                "Install a CuPy package matching your CUDA Toolkit.",
                UserWarning,
                stacklevel=2,
            )
            xp = np

    data = xp.asarray(dataset, dtype=float)

    if data.ndim == 1:
        data = data.reshape(1, -1)
        single_series = True
    elif data.ndim == 2:
        single_series = False
    else:
        raise ValueError("Only 1D or 2D arrays are allowed!")

    series_len = data.shape[1]
    if s_values is None:
        s_max = int(series_len / 4)
        if s_max < s_min:
            raise ValueError(
                f"Cannot create scales with s_max({s_max}) < s_min({s_min})"
            )
        s_values = [
            int(exp(step)) for step in np.arange(np.log(s_min), np.log(s_max), 0.5)
        ]
    elif isinstance(s_values, Sequence):
        init_s_len = len(s_values)
        if init_s_len < 1:
            raise ValueError("Input s_values is empty.")
        s_values = list(filter(lambda x: x <= series_len / 4, s_values))
        if len(s_values) < 1:
            raise ValueError(
                "Invalid s_values: all entries are greater than L / 4. "
                "No valid scales found for analysis."
            )

        if len(s_values) != init_s_len:
            warnings.warn(
                f"DFA warning: only following S values are in use: {s_values}"
                f"\nOriginal input had {init_s_len} values.",
                UserWarning,
                stacklevel=2,
            )

    elif isinstance(s_values, (float, int)):
        if s_values > series_len / 4:
            raise ValueError("Cannot use S > L / 4")
        s_values = (s_values,)

    n_series = data.shape[0]
    results = None

    if xp is not np and processes > 1:
        warnings.warn(
            "DFA warning: processes is ignored by the CUDA backend; "
            "using the current GPU from a single process.",
            UserWarning,
            stacklevel=2,
        )

    if processes <= 1 or xp is not np:
        indices = np.arange(n_series)
        results = dfa_worker(
            indices=indices,
            arr=data,
            degree=degree,
            s_values=s_values,
            n_integral=n_integral,
            xp=xp,
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
        f2_out = xp.vstack(f2_list)

    if xp is not np:
        f2_out = xp.asnumpy(f2_out)

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
        )
        scales, fluct2_values = result_list[0]

        # Convert to log format for backward compatibility
        log_s = np.log(scales)
        # Extract F(s) from F^2(s) and take logarithm
        log_f = np.log(np.sqrt(fluct2_values))

        return log_s, log_f

    def find_h(self, simple_mode: bool = True) -> Union[float, np.ndarray]:
        """
        Estimate the Hurst exponent from fluctuation analysis.

        External interface is preserved:
        - returns scalar for 1D input;
        - returns 1D array for 2D input.

        Args:
            simple_mode: Kept for backward compatibility, not used
                        (always uses linear fit).
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
        )
        self.s, self.F_s = scales, fluct2_values

        log_s = np.log(self.s)

        if self.dataset.ndim == 1:
            log_f = np.log(np.sqrt(self.F_s))
            slope = self._hurst_exponent(log_s, log_f, simple_mode=True)
            h_value = slope
        else:
            n_series = self.dataset.shape[0]
            h_array = np.empty(n_series, dtype=float)
            for i in range(n_series):
                log_f = np.log(np.sqrt(self.F_s[i]))
                slope = self._hurst_exponent(log_s, log_f, simple_mode=True)
                h_array[i] = slope
            h_value = h_array

        return h_value

    def parallel_2d(
        self,
        threads: int = cpu_count(),
        progress_bar: bool = False,
        h_control: bool = False,
        h_target: float = float(),
        h_limit: float = float(),
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

            # Normalize to 2D format (single row)
            vector_2d = vector.reshape(1, -1)

            result_list = dfa_worker(
                indices=0,
                arr=vector_2d,
                degree=self.degree,
                s_values=None,
                n_integral=1,
            )
            scales, fluct2_values = result_list[0]

            log_s = np.log(scales)
            log_f = np.log(np.sqrt(fluct2_values))
            slope = self._hurst_exponent(log_s, log_f, simple_mode=True)
            h_calc = slope

            np.frombuffer(estimations.get_obj())[i] = h_calc

            with shared_lock:
                shared_counter.value += 1

            if h_control and abs(h_calc - h_target) > h_limit:
                invalid_i.append(i)

        return np.array(invalid_i)
