import gc
from collections.abc import Iterable
from contextlib import closing
from ctypes import c_double
from functools import partial
from multiprocessing import Pool
from typing import Union

import numpy as np

from StatTools.auxiliary import SharedBuffer


class DPCCA:
    """
    Detrended Partial Cross-Correlation Analysis (DPCCA) implementation.

    DPCCA is an extension of Detrended Cross-Correlation Analysis (DCCA) that
    examines the partial cross-correlations between two time series while
    controlling for the influence of other time series. It provides insights
    into the intrinsic relationships between time series by removing spurious
    correlations.

    The method computes:
    - P: Partial cross-correlation coefficients on different time scales
    - R: Cross-correlation coefficients (may include spurious correlations)
    - F: Covariance matrix of detrended fluctuations

    Basic usage:
        ```python
        import numpy as np
        from StatTools.analysis.dpcca_as_class import DPCCA

        # Multiple time series data (shape: n_series x length)
        data = np.random.normal(0, 1, (5, 10000))

        # Create DPCCA analyzer
        dpcca = DPCCA(data, pd=2, step=0.5, s=[16, 32, 64, 128])

        # Perform analysis
        P, R, F, scales = dpcca.forward()

        # P[i,j,s] contains partial cross-correlation between series i,j at scale s
        ```

    Args:
        arr (numpy.ndarray or SharedBuffer): Input time series data.
            Shape should be (n_series, length) for multiple series.
        pd (int): Polynomial degree for detrending (default: 2)
        step (float): Step size for scale segmentation (0 < step <= 1)
        s (Union[int, Iterable]): Scale values for analysis. Can be:
            - Single integer for one scale
            - List/array of integers for multiple scales

    Attributes:
        arr (numpy.ndarray or SharedBuffer): Input data
        pd (int): Polynomial degree for detrending
        step (float): Step size for segmentation
        s (Union[int, Iterable]): Scale values
        shape (tuple): Shape of input data

    Raises:
        ValueError: If step is not in (0, 1] or if scale values are invalid
        TypeError: If input types are incorrect
    """

    def __init__(
        self,
        arr: Union[np.ndarray, SharedBuffer],
        pd: int,
        step: float,
        s: Union[int, Iterable],
    ):
        """
        Initialize DPCCA analyzer.

        Args:
            arr: Input time series data
            pd: Polynomial degree for detrending
            step: Step size for scale segmentation
            s: Scale values for analysis
        """
        if isinstance(arr, np.ndarray):
            self.arr = [arr] if arr.ndim == 1 else arr
        else:
            self.arr = arr
        self.pd, self.step, self.s = pd, step, s
        self.shape = arr.shape

        if not 0 < step <= 1:
            raise ValueError("0 < step <= 1 !")

    def forward(self, processes: int = 1, force_gc: Union[bool, tuple] = False):
        """
        Execute DPCCA analysis.

        Performs the complete DPCCA analysis on the input data, computing
        partial cross-correlations, cross-correlations, and fluctuation
        covariances at specified scales.

        Args:
            processes (int): Number of parallel processes (default: 1)
            force_gc (Union[bool, tuple]): Garbage collection control.
                - False: No forced GC
                - True: Force GC every 2 cycles
                - tuple: (gc_frequency, gc_generations)

        Returns:
            tuple: (P, R, F, s) where:
                - P: Partial cross-correlation matrix (shape: n_series x n_series x n_scales)
                - R: Cross-correlation matrix (shape: n_series x n_series x n_scales)
                - F: Fluctuation covariance matrix (shape: n_series x n_series x n_scales)
                - s: Scale values used in analysis

        Raises:
            ValueError: If scale values exceed data length constraints
            TypeError: If scale input type is invalid
        """
        if force_gc:
            force_gc = (2, 2)

        if isinstance(self.s, (tuple, list, np.ndarray)):
            init_s_len = len(self.s)

            s = list(filter(lambda x: x <= self.shape[1] / 4, self.s))
            if len(s) < 1:
                raise ValueError(
                    "All input S values are larger than vector shape / 4 !"
                )

            if len(s) != init_s_len:
                print(f"\tDPCAA warning: only following S values are in use: {s}")

            processes = len(s) if processes > len(s) else processes

            S = np.array(s, dtype=int) if not isinstance(self.s, np.ndarray) else s
            S_by_workers = np.array_split(S, processes)

            if processes == 1:
                return self._dpcca_worker(s) + s

            if isinstance(self.arr, np.ndarray):
                chunk = SharedBuffer(self.shape, c_double)
                chunk.write(self.arr)
                chunk.apply_in_place(np.cumsum, by_1st_dim=True)
                self.arr = chunk
            elif isinstance(self.arr, SharedBuffer):
                self.arr.apply_in_place(np.cumsum, by_1st_dim=True)

            with closing(
                Pool(
                    processes=processes,
                    initializer=self.arr.buffer_init,
                    initargs=({"ARR": self.arr},),
                )
            ) as pool:
                result = pool.map(
                    partial(self._dpcca_worker, force_gc=force_gc), S_by_workers
                )

        elif isinstance(self.s, int):
            if self.s > self.shape[1] / 4:
                raise ValueError("Cannot use S > L / 4")
        else:
            raise TypeError(
                "Input S values could be : int, tuple, list or numpy.ndarray!"
            )

    def _dpcca_worker(self, s: Union[int, Iterable], force_gc: Union[bool, tuple]):
        """
        Core DPCCA computation worker.

        Performs the actual DPCCA calculations for given scale values.
        This method implements the mathematical core of the DPCCA algorithm.

        Args:
            s: Scale value(s) to compute correlations for
            force_gc: Garbage collection control parameters

        Returns:
            tuple: (P, R, F) matrices for the computed scales where:
                - P: Partial cross-correlation coefficients
                - R: Cross-correlation coefficients
                - F: Fluctuation covariances
        """
        s_current = [s] if not isinstance(s, Iterable) else s

        cumsum_arr = (
            SharedBuffer.get("ARR")
            if isinstance(self.arr, SharedBuffer)
            else np.cumsum(self.arr, axis=1)
        )

        shape = self.arr.shape

        F = np.zeros((len(s_current), shape[0], shape[0]), dtype=float)
        R = np.zeros((len(s_current), shape[0], shape[0]), dtype=float)
        P = np.zeros((len(s_current), shape[0], shape[0]), dtype=float)

        for s_i, s_val in enumerate(s_current):

            V = np.arange(0, shape[1] - s_val, int(self.step * s_val))
            Xw = np.arange(s_val, dtype=int)
            Y = np.zeros((shape[0], len(V)), dtype=object)

            for n in range(cumsum_arr.shape[0]):
                for v_i, v in enumerate(V):
                    W = cumsum_arr[n][v : v + s_val]
                    if len(W) == 0:
                        print(f"\tFor s = {s_val} W is an empty slice!")
                        return P, R, F

                    p = np.polyfit(Xw, W, deg=self.pd)
                    Z = np.polyval(p, Xw)
                    Y[n][v_i] = Z - W

                    if isinstance(force_gc, tuple):
                        if n % force_gc[0] == 0:
                            gc.collect(force_gc[1])

            Y = np.array([np.concatenate(Y[i]) for i in range(Y.shape[0])])

            for n in range(shape[0]):
                for m in range(n + 1):
                    F[s_i][n][m] = np.mean(Y[n] * Y[m])  # / (s_val - 1)
                    F[s_i][m][n] = F[s_i][n][m]

            for n in range(shape[0]):
                for m in range(n + 1):
                    R[s_i][n][m] = F[s_i][n][m] / np.sqrt(F[s_i][n][n] * F[s_i][m][m])
                    R[s_i][m][n] = R[s_i][n][m]

            Cinv = np.linalg.inv(R[s_i])

            for n in range(shape[0]):
                for m in range(n + 1):
                    if Cinv[n][n] * Cinv[m][m] < 0:
                        print(
                            f"S = {s_val} | Error: Sqrt(-1)! No P array values for this S!"
                        )
                        break

                    P[s_i][n][m] = -Cinv[n][m] / np.sqrt(Cinv[n][n] * Cinv[m][m])
                    P[s_i][m][n] = P[s_i][n][m]
                else:
                    continue
                break

        return P, R, F
