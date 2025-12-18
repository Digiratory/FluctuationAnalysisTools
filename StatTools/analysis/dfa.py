import time
from contextlib import closing
from ctypes import c_double
from functools import partial
from math import exp, floor
from multiprocessing import Pool, cpu_count, Array, Lock, Value
from typing import Union, Tuple
from tqdm import TqdmWarning, tqdm
from threading import Thread

import numpy as np
from StatTools.auxiliary import SharedBuffer


# ====================== Вспомогательные функции DFA ======================

def _detrend_segment(indices: np.ndarray, y_segment: np.ndarray, degree: int) -> float:
    """
    Detrending of a single data segment. Fits a polynomial and calculates
    the mean squared residual (F^2).
    """
    coef = np.polyfit(indices, y_segment, deg=degree)
    current_trend = np.polyval(coef, indices)
    f2 = np.sum(np.power((y_segment - current_trend), 2)) / len(y_segment)
    return f2


def _fluctuation_function(f_q_s_sum: float, cycles_amount: int, degree: int, segment_length: int,
                          root: bool = False) -> float:
    """
    Calculation of the fluctuation function F(s) for a given scale.
    """
    f1 = np.power(((1 / cycles_amount) * f_q_s_sum), 1 / degree)
    if root:
        return f1 / np.sqrt(segment_length)
    else:
        return f1


def _hurst_exponent(x_axis: np.ndarray, y_axis: np.ndarray, simple_mode: bool = True) -> float:
    """
    Calculation of the Hurst exponent (scaling exponent) by fitting
    log(F(s)) vs log(s) with a linear regression.
    """
    if simple_mode:
        return np.polyfit(x_axis, y_axis, deg=1)[0]
    else:
        error_str = "\n    Non-linear approximation is non supported yet!"
        raise NameError(error_str)


# ====================== Основной воркер DFA ======================

def dfa_worker(
        indices: Union[int, list, np.ndarray],
        arr: Union[np.ndarray, None] = None,
        degree: int = 2,
        root: bool = False,
        buffer_in_use: bool = False,
        simple_mode: bool = True,
        s_values: Union[list, np.ndarray, None] = None,
) -> Union[Tuple[np.ndarray, np.ndarray, float], list]:
    """
    Core of the DFA algorithm. Processes a subset of series (indices) and
    returns fluctuation functions and scaling exponents.

    Args:
        indices: Indices of the time series in the dataset to process.
        arr: Dataset array (ignored if buffer_in_use is True).
        degree: Order of polynomial detrending.
        root: If True, normalizes F(s) by sqrt(s).
        buffer_in_use: Flag to use SharedBuffer for memory efficiency.
        simple_mode: Use linear fit for Hurst exponent calculation.
        s_values: Pre-calculated box sizes (scales).

    Returns:
        A list of tuples or a single tuple containing:
        [log_s, log_F, h] where 'h' is the calculated Hurst exponent.
    """
    if buffer_in_use:
        data = SharedBuffer.get("ARR").to_array()
    else:
        data = arr

    if not isinstance(indices, (list, np.ndarray)):
        indices = [indices]
        single_output = True
    else:
        single_output = False

    if s_values is None:
        series_len = len(data) if data.ndim == 1 else data.shape[1]
        s_max = int(series_len / 4)
        s_values = [int(exp(step)) for step in np.arange(1.6, np.log(s_max), 0.5)]

    results = []
    data_ndim = data.ndim

    for idx in indices:
        series = data if data_ndim == 1 else data[idx]

        # Standard DFA preprocessing: center and integrate
        data_mean = np.mean(series)
        data_centered = series - data_mean
        y_cumsum = np.cumsum(data_centered)
        series_len = len(data_centered)

        log_s_vals = []
        log_F_vals = []

        for s_val in s_values:
            if s_val >= series_len / 4: continue

            s = np.linspace(1, s_val, s_val, dtype=int)
            len_s = len(s)
            cycles_amount = floor(series_len / len_s)

            if cycles_amount < 1: continue

            f_q_s_sum = 0
            s_temp = s.copy()

            for i in range(1, cycles_amount):
                indices_s = np.array((s_temp - (i + 0.5) * len_s), dtype=int)
                y_cumsum_s = np.take(y_cumsum, s_temp)

                f2 = _detrend_segment(indices_s, y_cumsum_s, degree)
                f_q_s_sum += np.power(f2, (degree / 2))
                s_temp += s_val

            f1 = _fluctuation_function(f_q_s_sum, cycles_amount, degree, len_s, root)
            log_s_vals.append(np.log(s_val))
            log_F_vals.append(np.log(f1))

        log_s_vals = np.array(log_s_vals)
        log_F_vals = np.array(log_F_vals)

        h = _hurst_exponent(log_s_vals, log_F_vals, simple_mode)
        results.append((log_s_vals, log_F_vals, h))

    if single_output:
        return results[0]
    return results


# ====================== Вспомогательная функция запуска Pool ======================

def _run_in_pool(func, iterable, processes, buffer_array=None):
    """
    Universal process pool runner with optional SharedBuffer initialization.
    """
    pool_params = {}

    if buffer_array is not None:
        buffer_shape = buffer_array.shape
        if len(buffer_shape) == 1:
            buffer_shape = (1, buffer_shape[0])
            buffer_array = buffer_array.reshape(1, -1)

        shared_buffer = SharedBuffer(buffer_shape, c_double)
        shared_buffer.write(buffer_array)

        pool_params['initializer'] = shared_buffer.buffer_init
        pool_params['initargs'] = ({"ARR": shared_buffer},)

    with closing(Pool(processes=processes, **pool_params)) as pool:
        return pool.map(func, iterable)


# ====================== Управляющая функция DFA ======================

def dfa(
        dataset,
        degree: int = 2,
        root: bool = False,
        processes: int = 1,
        buffer: bool = False,
        simple_mode: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Union[float, np.ndarray]]:
    """
    Implementation of the Detrended Fluctuation Analysis (DFA) method.

    The algorithm removes local polynomial trends in integrated time series and
    analyzes the scaling of the remaining fluctuations.

    Args:
        dataset (ndarray): 1D or 2D array of time series data.
        degree (int): Polynomial degree for detrending (default: 2).
        root (bool): If True, fluctuation function is divided by sqrt(s).
        processes (int): Number of parallel workers (default: 1).
        buffer (bool): If True, uses SharedBuffer for IPC (recommended for large datasets).
        simple_mode (bool): If True, calculates a single Hurst exponent via linear fit.

    Returns:
        tuple: (log_s, log_F, Hurst_exponent)
            - For 1D input: Returns three individual values/arrays.
            - For 2D input: Returns arrays where each row corresponds to an input series.

    Raises:
        ValueError: If input dimension is not 1D or 2D.
    """
    dataset = np.array(dataset)
    if dataset.ndim > 2 or dataset.ndim == 0:
        raise ValueError("Only 1D or 2D arrays are allowed!")

    series_len = len(dataset) if dataset.ndim == 1 else dataset.shape[1]
    s_max = int(series_len / 4)
    s_values = [int(exp(step)) for step in np.arange(1.6, np.log(s_max), 0.5)]

    # Sequential execution
    if not processes > 1:
        indices = 0 if dataset.ndim == 1 else list(range(dataset.shape[0]))
        result = dfa_worker(
            indices, arr=dataset, degree=degree, root=root,
            buffer_in_use=False, simple_mode=simple_mode, s_values=s_values
        )
        if dataset.ndim == 2:
            return (
                np.array([r[0] for r in result]),
                np.array([r[1] for r in result]),
                np.array([r[2] for r in result])
            )
        return result

    # Parallel execution
    processes = min(processes, cpu_count())
    worker_args = {
        'degree': degree, 'root': root,
        'buffer_in_use': buffer, 'simple_mode': simple_mode
    }

    if dataset.ndim == 1:
        # 1D Case: Parallelize by scales (S-values)
        processes = min(processes, len(s_values))
        chunks = np.array_split(s_values, processes)

        worker_func = partial(dfa_worker, indices=0, **worker_args, arr=None if buffer else dataset)
        task_func = lambda s_chunk: worker_func(s_values=s_chunk)

        results = _run_in_pool(task_func, chunks, processes, buffer_array=dataset if buffer else None)

        all_log_s = np.concatenate([r[0] for r in results])
        all_log_F = np.concatenate([r[1] for r in results])
        h_total = _hurst_exponent(all_log_s, all_log_F, simple_mode)

        return all_log_s, all_log_F, h_total

    else:
        # 2D Case: Parallelize by series (rows)
        n_series = dataset.shape[0]
        processes = min(processes, n_series)
        chunks = np.array_split(np.arange(n_series), processes)

        task_func = partial(
            dfa_worker,
            arr=None if buffer else dataset,
            s_values=s_values,
            **worker_args
        )

        results_list_of_lists = _run_in_pool(task_func, chunks, processes, buffer_array=dataset if buffer else None)
        flat_results = [item for sublist in results_list_of_lists for item in sublist]

        return (
            np.array([r[0] for r in flat_results]),
            np.array([r[1] for r in flat_results]),
            np.array([r[2] for r in flat_results])
        )

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


# ====================== КЛАСС DFA (ОБЁРТКА) ======================
class DFA:
    """
    Detrended Fluctuation Analysis (DFA) implementation for estimating Hurst exponent.
    """

    def __init__(self, dataset, degree=2, root=False, ignore_input_control=False):
        self.degree = degree
        self.root = root
        self.dataset = None
        self.s = None
        self.F_s = None

        if ignore_input_control:
            result = dfa_worker(0, arr=dataset, degree=degree, root=root, buffer_in_use=False, simple_mode=True)
            self.s, self.F_s, _ = result
        else:
            if isinstance(dataset, str):
                try:
                    dataset = np.loadtxt(dataset)
                except OSError:
                    raise NameError("\n    The file either doesn't exit or you use wrong path!")
                if np.size(dataset) == 0:
                    raise NameError("\n    Input file is empty!")

            if not isinstance(dataset, np.ndarray):
                try:
                    dataset = np.array(dataset, dtype=float)
                except (ValueError, TypeError):
                    raise NameError("\n    Input dataset is supposed to be numpy array, list or directory!")

            if dataset.ndim > 2 or dataset.ndim == 0:
                raise NameError("\n    Only 1- or 2-dimensional arrays are allowed!")

            series_len = len(dataset) if dataset.ndim == 1 else dataset.shape[1]
            if series_len < 20:
                raise NameError("Wrong input array ! (It's probably too short)")

            self.dataset = np.array(dataset)

    @staticmethod
    def initializer_for_parallel_mod(shared_array, h_est, shared_c, shared_l):
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
    def dfa_core_cycle(dataset, degree, root):
        result = dfa_worker(0, arr=dataset, degree=degree, root=root, buffer_in_use=False, simple_mode=True)

        return result[0], result[1]

    def find_h(self, simple_mode=True):
        """
        Estimate Hurst exponent from fluctuation analysis.
        """
        if not simple_mode:
            error_str = "\n    Non-linear approximation is not supported yet!"
            raise NameError(error_str)

        result = dfa(self.dataset, self.degree, self.root, processes=1, buffer=False, simple_mode=True)
        self.s, self.F_s, h_value = result
        return h_value

    def parallel_2d(
            self,
            threads=cpu_count(),
            progress_bar=False,
            h_control=False,
            h_target=float(),
            h_limit=float(),
    ):

        if threads == 1 or self.dataset.ndim == 1:
            return self.find_h()

        if len(self.dataset) / threads < 1:
            print(
                "\n    DFA Warning: Input array is too small for using it in parallel mode!"
                f"\n    You better use either less threads ({len(self.dataset)}) or don't use parallel mode!"
            )
            return self.find_h()

        vectors_indices_by_threads = np.array_split(
            np.linspace(0, len(self.dataset) - 1, len(self.dataset), dtype=int),
            threads,
        )

        dataset_to_memory = Array(c_double, len(self.dataset) * len(self.dataset[0]))
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
            # map вызывает parallel_core, который теперь обёртка над dfa_worker
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
            return np.frombuffer(h_estimation_in_memory.get_obj()), invalid_i
        else:
            return np.frombuffer(h_estimation_in_memory.get_obj())

    def parallel_core(self, indices, quantity, length, h_control, h_target, h_limit):

        invalid_i = []

        all_data = np.frombuffer(datasets_array.get_obj()).reshape((quantity, length))

        for i in indices:
            vector = all_data[i]
            result = dfa_worker(
                indices=0,
                arr=vector,
                degree=self.degree,
                root=self.root,
                buffer_in_use=False,
                simple_mode=True
            )

            h_calc = result[2]

            np.frombuffer(estimations.get_obj())[i] = h_calc

            with shared_lock:
                shared_counter.value += 1

            if h_control:
                if abs(h_calc - h_target) > h_limit:
                    invalid_i.append(i)

        return np.array(invalid_i)
