import time
from contextlib import closing
from ctypes import c_double
from functools import partial
from multiprocessing import Array, Pool, Value, cpu_count
from threading import Thread

from numpy import (
    arange,
    array,
    array_split,
    copyto,
    frombuffer,
    linspace,
    ndarray,
    round,
    zeros,
)
from numpy.random.mtrand import normal
from tqdm import TqdmWarning, tqdm

from StatTools import StatTools_bindings


class QSS:
    """
    QSS stands for queueing simulation system.

    It let us generate poisson thread based on input vector and then
    acquire a corresponding average waiting time.

    For more information: https://sciencedirect.com/science/article/abs/pii/S0378437117305344

    Usage: the are two ways to work process input data

    1. Simply getting a waiting time for your dataset:

        x = numpy.random.normal(100, 25, (100, 1440))           # your data
        waiting_curves = QSS(x, U=[0.5, 0.6, 0.7, 0.8, 0.9], threads=1, progress_bar=True).compute_waiting()

    2. When multiple calls are required we can deploy pool of processes only once and then reuse it. In
    this case the processes' initialization overhead is smaller which can be really useful when computing
    waiting curves for a large dataset divided in smaller chunks.

    The following pattern can be used:

        sim = QSS(x, u=U, threads=12, progress_bar=True)        # first we need to deploy the pool
        shape = (100, 1440)                                     # we need to declare in advance what shape out data
                                                                is going to have during computation. It has to have
                                                                same size for each iteration.

        for i in range(5):                                      # suppose we've got 5 independent chunks of data
            x = normal(200, 20, shape)                          # sample data with given shape
            sim.rewrite_array(x)                                # here we declare out data to be shared between cpu
            curves = sim.compute_waiting(multiple_runs=True)    # computation itself

        sim.stop()                                              # IT'S NECESSARY TO EXPLICITLY TERMINATE OUT POOL!

    """

    def __init__(
        self,
        data: (ndarray, tuple),
        u: (ndarray, list, tuple),
        threads=cpu_count(),
        progress_bar=True,
    ):

        self._controller = Value("i", 0)
        self._bar_val = Value("i", 0, lock=True)

        if isinstance(u, (list, tuple)):
            u = array(u)

        self.U = u

        if isinstance(data, ndarray):
            self._input_shape = (len(data), len(data[0]))
            self._output_shape = (len(data), len(self.U))
        elif isinstance(data, tuple):
            self._input_shape = data
            self._output_shape = (self._input_shape[0], len(self.U))

        self._shared_input = Array(
            c_double, self._input_shape[0] * self._input_shape[1], lock=True
        )
        self._shared_output = Array(
            c_double, self._output_shape[0] * self._output_shape[1], lock=True
        )
        if isinstance(data, ndarray):
            copyto(
                frombuffer(self._shared_input.get_obj(), dtype=c_double).reshape(
                    self._input_shape
                ),
                data,
            )

        self._threads, self._progress_bar = threads, progress_bar

        if self._threads > 1:
            self._pool = Pool(
                processes=self._threads,
                initializer=self._global_init,
                initargs=(
                    self._shared_input,
                    self._shared_output,
                    self._controller,
                    self._bar_val,
                ),
            )
            if self._progress_bar:
                Thread(
                    target=self._bar_manager, args=(self._input_shape[0], self._bar_val)
                ).start()

    def compute_waiting(self, C0=None, multiple_runs=False):

        indices = array_split(
            linspace(0, self._input_shape[0] - 1, self._input_shape[0], dtype=int),
            self._threads,
        )

        if self._threads == 1:
            self._global_init(
                self._shared_input, self._shared_output, self._controller, self._bar_val
            )
            self._worker(
                indices[0],
                C0,
                self.U,
                self._input_shape,
                self._output_shape,
                self._progress_bar,
                True,
            )
            return self.get_result()
        else:
            if self._progress_bar:
                self._bar_val.value = 0
                Thread(
                    target=self._bar_manager, args=(self._input_shape[0], self._bar_val)
                ).start()

            self._pool.map_async(
                partial(
                    self._worker,
                    C0=C0,
                    U=self.U,
                    input_shape=self._input_shape,
                    output_shape=self._output_shape,
                    bar_on=self._progress_bar,
                    linear=False,
                ),
                indices,
            )

            while True:
                time.sleep(0.2)
                if self._controller.value == self._threads:
                    if not multiple_runs:
                        self._pool.terminate()
                    self._controller.value = 0
                    self._bar_val.value = -1
                    return self.get_result()

    def stop(self):
        self._pool.terminate()

    def rewrite_array(self, arr):
        copyto(
            frombuffer(self._shared_input.get_obj(), dtype=c_double).reshape(
                self._input_shape
            ),
            arr,
        )
        copyto(
            frombuffer(self._shared_output.get_obj(), dtype=c_double).reshape(
                self._output_shape
            ),
            zeros(self._output_shape, dtype=float),
        )

    def get_result(self):
        return frombuffer(self._shared_output.get_obj(), dtype=c_double).reshape(
            self._output_shape
        )

    @staticmethod
    def _worker(indices, C0, U, input_shape, output_shape, bar_on, linear):
        def get_vector(i):
            return frombuffer(SHARED_INPUT.get_obj(), dtype=c_double).reshape(
                input_shape
            )[i]

        def write_result_to_mem(i, vec):
            frombuffer(SHARED_OUTPUT.get_obj(), dtype=c_double).reshape(output_shape)[
                i
            ] = vec

        bar = None

        if linear:
            bar = tqdm(desc="C_QSS", total=len(indices), disable=not bar_on)

        c = -1

        for i in indices:

            if C0 is not None:
                if isinstance(C0, (int, float)):
                    c = C0
                elif isinstance(C0, (ndarray, list, tuple)):
                    c = C0[i]

            waiting_curve = StatTools_bindings.get_waiting_time(get_vector(i), U, c)
            write_result_to_mem(i, waiting_curve)
            if linear:
                bar.update(1)
            else:
                BAR.value += 1

        with CONTROLLER.get_lock():
            CONTROLLER.value += 1

        if linear:
            bar.close()

    @staticmethod
    def _global_init(sh_input, sh_output, c_val, b_val):
        global SHARED_INPUT
        global SHARED_OUTPUT
        global CONTROLLER
        global BAR
        SHARED_INPUT = sh_input
        SHARED_OUTPUT = sh_output
        CONTROLLER = c_val
        BAR = b_val

    def _bar_manager(self, total, counter):

        with closing(tqdm(desc="Progress", total=100, leave=False, position=0)) as bar:
            try:
                last_val = counter.value
                while True:
                    if counter.value == -1:
                        break

                    time.sleep(0.25)
                    with counter.get_lock():
                        if counter.value > last_val:
                            v = round((counter.value - last_val) * 100 / total, 2)
                            bar.update(v)
                            last_val = counter.value

                        if counter.value == total:
                            bar.close()
                            break
            except TqdmWarning:
                return


if __name__ == "__main__":

    shape = (1200, 1440)

    x = normal(100, 20, shape)

    U = arange(0.5, 1.0, 0.1)
    sim = QSS(x, u=[0.5, 0.9], threads=12, progress_bar=True).compute_waiting()
    print(sim[0:5])

    """for i in range(5):
        x = normal(200, 20, shape)
        sim.rewrite_array(x)
        curves = sim.compute_waiting(multiple_runs=True)
        time.sleep(1)
        print(curves[0])

    sim.stop()"""
