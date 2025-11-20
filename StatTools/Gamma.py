from contextlib import closing
from ctypes import c_double
from functools import partial
from multiprocessing import Array, Lock, Pool, Value
from threading import Thread

import numpy
from tqdm import tqdm

from StatTools.analysis.dfa import bar_manager


class GammaReplacement:
    """
    Transform datasets to follow Gamma distribution using rank replacement.

    This class implements a statistical transformation technique that converts
    any input distribution to a Gamma distribution while preserving the rank
    order of the original data. The method works by:

    1. Sorting the original data
    2. Generating a Gamma-distributed sample with target parameters
    3. Sorting the Gamma sample
    4. Replacing original values with corresponding Gamma values by rank

    This preserves the ordinal relationships in the data while imposing
    the desired Gamma distribution characteristics.

    Basic usage:
        ```python
        import numpy as np
        from StatTools.Gamma import GammaReplacement

        # Create sample data with arbitrary distribution
        data = np.random.normal(0, 1, (100, 1000))

        # Transform to Gamma distribution with mean=20, std=6
        gamma_transformer = GammaReplacement(data, set_mean=20, set_std=6)
        transformed_data = gamma_transformer.perform(threads=4, progress_bar=True)
        ```

    Args:
        dataset (array-like): Input data to transform. Can be 1D or 2D array.
        set_mean (float): Target mean for Gamma distribution (default: 20)
        set_std (float): Target standard deviation for Gamma distribution (default: 6)

    Attributes:
        dataset (numpy.ndarray): Input data converted to numpy array
        set_mean (float): Target mean value
        set_std (float): Target standard deviation

    Raises:
        NameError: If set_mean is negative or either set_mean/set_std is zero

    Note:
        The transformation preserves the rank order and relative relationships
        of the original data while imposing Gamma distribution properties.
        Useful for statistical modeling and simulation studies.
    """

    def __init__(self, dataset, set_mean=20, set_std=6):

        if set_mean < 0:
            error_str = (
                "\tGammaReplacement Error: Mean value is not supposed to be negative!"
            )
            raise NameError(error_str)

        if set_std == 0 or set_mean == 0:
            error_str = "\tGammaReplacement Error: Zero mean or std!"
            raise NameError(error_str)

        if not isinstance(dataset, type(numpy.array([]))):
            self.dataset = numpy.array(dataset)

        else:
            self.dataset = dataset

        self.set_std = set_std
        self.set_mean = set_mean

    @staticmethod
    def rank_replacement_with_gamma(initial_vector):
        """
        Perform rank-preserving transformation to Gamma distribution.

        Transforms an input vector to follow a Gamma distribution while maintaining
        the rank order of the original values. This is achieved by:

        1. Estimating Gamma parameters (shape k, scale θ) from the input data
        2. Generating a Gamma-distributed sample with these parameters
        3. Replacing original values with Gamma values based on rank order

        Args:
            initial_vector (numpy.ndarray): Input vector to transform

        Returns:
            numpy.ndarray: Transformed vector following Gamma distribution
                          with preserved rank order

        Note:
            The Gamma parameters are estimated using method of moments:
            - θ (scale) = variance / mean
            - k (shape) = mean / θ
        """
        sorted_x1 = numpy.sort(initial_vector)

        var_normal = numpy.var(initial_vector, ddof=1)
        mean_normal = numpy.mean(initial_vector)

        theta = var_normal / mean_normal
        k = mean_normal / (var_normal / mean_normal)
        validation_vector = numpy.random.mtrand.gamma(k, theta, len(initial_vector))

        sorted_x2 = numpy.sort(validation_vector)

        # @njit(cache=True)
        def rank_replacement_core(x1, sorted_x1, sorted_x2):
            for i in range(len(x1)):
                get_index_in_sorted_x1 = numpy.where(sorted_x1 == x1[i])[0][0]
                x1[i] = sorted_x2[get_index_in_sorted_x1]
            return x1

        x1 = rank_replacement_core(initial_vector, sorted_x1, sorted_x2)

        return x1

    def perform(self, threads=1, progress_bar=False):
        """
        Execute the Gamma distribution transformation.

        Performs the complete transformation process on the input dataset,
        converting each time series to follow a Gamma distribution with
        specified mean and standard deviation parameters.

        The transformation process for each vector:
        1. Scale to target standard deviation
        2. Shift to target mean
        3. Apply rank-preserving Gamma transformation
        4. Re-scale and re-center to ensure exact target statistics

        Args:
            threads (int): Number of parallel threads for processing (default: 1).
                Recommended for large datasets (>10^6 elements).
            progress_bar (bool): Show progress bar during processing (default: False)

        Returns:
            numpy.ndarray: Transformed dataset with same shape as input.
                Each time series follows Gamma distribution with target parameters.

        Note:
            - Single-threaded processing for small datasets
            - Multi-threaded processing with shared memory for large datasets
            - Progress bar uses tqdm for visual feedback
            - Automatic warnings for suboptimal thread usage
        """

        if threads == 1:

            if self.dataset.size > pow(10, 6):
                print(
                    "\tGamaReplacement Warning: Given the size of input dataset it'd be faster to use more threads . . ."
                )

            if progress_bar:
                bar = tqdm(
                    desc="GammaReplacment",
                    total=len(self.dataset),
                    leave=False,
                    position=0,
                )

            one_dim = False

            if self.dataset.ndim == 1:

                self.dataset = [self.dataset]
                one_dim = True

            for v, vector in enumerate(self.dataset):
                self.dataset[v] = self.dataset[v] * (
                    self.set_std / numpy.std(self.dataset[v], ddof=1)
                )
                self.dataset[v] = self.dataset[v] + (
                    -numpy.mean(self.dataset[v]) + self.set_mean
                )
                self.dataset[v] = self.rank_replacement_with_gamma(self.dataset[v])
                self.dataset[v] = self.dataset[v] * (
                    self.set_std / numpy.std(self.dataset[v], ddof=1)
                )
                self.dataset[v] = self.dataset[v] + (
                    -numpy.mean(self.dataset[v]) + self.set_mean
                )
                a = numpy.std(self.dataset[v], ddof=1)
                if progress_bar:
                    bar.update(1)

            if one_dim:
                return self.dataset[0]

            return self.dataset

        else:

            vectors_indices = numpy.linspace(
                0, len(self.dataset) - 1, len(self.dataset), dtype=int
            )
            vectors_by_threads = numpy.array_split(vectors_indices, threads)

            shared_initial_dataset = Array(c_double, self.dataset.size)
            vectors_num = len(self.dataset)
            vectors_length = len(self.dataset[0])

            numpy.copyto(
                numpy.frombuffer(shared_initial_dataset.get_obj()).reshape(
                    (vectors_num, vectors_length)
                ),
                self.dataset,
            )

            bar_value = Value("i", 0)
            stop_bit = Value("i", 0)
            lock = Lock()

            if progress_bar:
                bar = Thread(
                    target=bar_manager,
                    args=(
                        "GammaReplacement",
                        vectors_num,
                        bar_value,
                        lock,
                        "total",
                        stop_bit,
                    ),
                )
                bar.start()

            with closing(
                Pool(
                    processes=threads,
                    initializer=self.global_initializer,
                    initargs=(shared_initial_dataset, bar_value, lock),
                )
            ) as pool:
                pool.map(
                    partial(
                        self.processing_vectors,
                        set_mean=self.set_mean,
                        set_std=self.set_std,
                        vectors_num=vectors_num,
                        vectors_length=vectors_length,
                        function=self.rank_replacement_with_gamma,
                    ),
                    vectors_by_threads,
                )

            stop_bit.value += 1

            output_array = numpy.frombuffer(shared_initial_dataset.get_obj()).reshape(
                (vectors_num, vectors_length)
            )

            return output_array

    @staticmethod
    def global_initializer(shared_init_data, b_val, b_lk):
        """
        Initialize global variables for parallel processing.

        Sets up shared memory pointers and synchronization objects
        for multiprocessing worker functions.

        Args:
            shared_init_data: Shared memory array containing dataset
            b_val: Shared counter for progress tracking
            b_lk: Lock for thread-safe progress updates
        """
        global SHARED_VECTORS
        global BAR_VALUE
        global BAR_LOCK
        SHARED_VECTORS = shared_init_data
        BAR_VALUE = b_val
        BAR_LOCK = b_lk

    @staticmethod
    def processing_vectors(
        indices, set_mean, set_std, vectors_num, vectors_length, function
    ):
        """
        Process a subset of vectors in parallel for Gamma transformation.

        Applies the complete Gamma transformation pipeline to a subset of
        time series vectors, updating the shared memory buffer with results.

        Args:
            indices (numpy.ndarray): Indices of vectors to process
            set_mean (float): Target mean for transformation
            set_std (float): Target standard deviation for transformation
            vectors_num (int): Total number of vectors in dataset
            vectors_length (int): Length of each vector
            function: Transformation function to apply (rank_replacement_with_gamma)

        Note:
            Updates shared memory buffer in-place and tracks progress
            using global counter with thread-safe increments.
        """

        def get_pointer():
            return numpy.frombuffer(SHARED_VECTORS.get_obj(), dtype=c_double).reshape(
                (vectors_num, vectors_length)
            )

        for v in indices:
            vector = get_pointer()[v]
            vector = vector * (set_std / numpy.std(vector, ddof=1))
            vector = vector + numpy.abs(numpy.mean(vector) - set_mean)
            vector = function(vector)
            vector = vector * (set_std / numpy.std(vector, ddof=1))
            vector = vector + numpy.abs(numpy.mean(vector) - set_mean)
            get_pointer()[v] = vector

            with BAR_LOCK:
                BAR_VALUE.value += 1
