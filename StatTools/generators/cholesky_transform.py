from collections.abc import Iterable
from multiprocessing import cpu_count
from typing import Union

import pandas as pd
from numpy import full, matmul, mean, ndarray
from numpy.linalg import cholesky

from StatTools.analysis.dfa import DFA
from StatTools.generators.base_filter import FilteredArray


class CorrelatedArray:
    """
    Generate mutually correlated time series using Cholesky decomposition.

    This class implements the Cholesky transformation method to create correlated
    time series from uncorrelated data. It uses the Cholesky decomposition of
    a target correlation matrix to transform independent random vectors into
    correlated ones while preserving their statistical properties.

    The Cholesky method works by:
    1. Creating a target correlation matrix R with desired correlations
    2. Computing the Cholesky decomposition L such that R = L * L^T
    3. Transforming uncorrelated data X as Y = L * X to get correlated data Y

    Basic usage with existing data:
        ```python
        import numpy as np
        from StatTools.generators.cholesky_transform import CorrelatedArray

        # Create uncorrelated dataset
        data = np.random.normal(0, 1, (100, 1024))

        # Generate correlated vectors with target correlation 0.8
        corr_generator = CorrelatedArray(data=data, threads=4)
        correlated_data = corr_generator.create(corr_target=0.8, h_control=False)
        ```

    Basic usage with generated data:
        ```python
        # Generate correlated data with Hurst exponent H=1.5
        corr_generator = CorrelatedArray(
            h=1.5, quantity=100, length=1024,
            set_mean=0, set_std=1, threads=4
        )
        correlated_data = corr_generator.create(corr_target=0.8, h_control=False)
        ```

    Args:
        h (float, optional): Hurst exponent for generating base data if no data provided
        quantity (int, optional): Number of time series to generate
        length (int, optional): Length of each time series
        data (ndarray, optional): Existing dataset to transform. If provided,
            h, quantity, and length are inferred from the data shape.
        set_mean (float): Mean value for generated data (default: 0)
        set_std (float): Standard deviation for generated data (default: 1)
        threads (int): Number of threads for parallel processing (default: CPU count)

    Attributes:
        dataset (ndarray): Input or generated uncorrelated data
        quantity (int): Number of time series
        length (int): Length of each time series
        set_mean (float): Target mean for generated data
        set_std (float): Target standard deviation for generated data
        threads (int): Number of processing threads

    Returns:
        pandas.DataFrame: Correlated time series data. Each column represents
            one time series. Index contains the target correlation coefficient.

    Notes:
        - **h_control**: When enabled, adds a column with DFA-estimated Hurst
          exponents for quality control of the generated data.
        - **Multiple correlations**: Pass a list of correlation targets to generate
          multiple correlation levels in a single call.
        - **Correlation guarantee**: The method produces vectors with approximately
          the target correlation. Perfect correlation matching is not guaranteed
          and may require post-processing selection.
        - **Performance**: Uses parallel processing for large datasets when threads > 1.

    Raises:
        TypeError: If required parameters are missing when no data is provided
    """

    def __init__(
        self,
        h=None,
        quantity=None,
        length=None,
        data=None,
        set_mean=0,
        set_std=1,
        threads=cpu_count(),
    ):

        if data is not None:
            self.length = length if length is not None else len(data[0])
            self.quantity = quantity if quantity is not None else len(data)
            self.dataset = data
        else:
            if length is None or quantity is None:
                raise TypeError(
                    f"Didn't specify {'length' if length is None else 'quantity'} . . ."
                )

            self.dataset = FilteredArray(h, length, set_mean, set_std).generate(
                n_vectors=quantity, progress_bar=False, threads=threads
            )
            self.quantity, self.length = quantity, length

        self.set_mean, self.set_std, self.threads = set_mean, set_std, threads

    def create(
        self, corr_target: Union[float, Iterable, ndarray], h_control=False
    ) -> pd.DataFrame:
        """
        Generate correlated time series using Cholesky transformation.

        This method implements the core Cholesky correlation transformation algorithm:
        1. Creates a target correlation matrix R with specified correlation coefficients
           (diagonal elements are 1.0, off-diagonal elements are corr_target)
        2. Computes the Cholesky decomposition L such that R = L * L^T
        3. Transforms the uncorrelated input data X as Y = L * X

        The resulting time series Y will have approximately the target correlation
        structure while preserving the original statistical properties of the input data.

        Args:
            corr_target (Union[float, Iterable, ndarray]): Target correlation coefficient(s).
                Can be:
                - Single float: Apply same correlation to all pairs
                - List/array of floats: Generate multiple correlation levels
                - Full correlation matrix: Use custom correlation structure
            h_control (bool): Enable Hurst exponent quality control. When True,
                performs DFA analysis on generated data and adds H_est column
                to the output DataFrame for validation.

        Returns:
            pd.DataFrame: Correlated time series data with the following structure:
                - Columns: Individual time series (0, 1, 2, ..., n-1)
                - Index: Target correlation coefficient for each row
                - Additional column "H_est" (if h_control=True): DFA-estimated Hurst exponents

        Raises:
            ValueError: If correlation matrix is not positive definite
            RuntimeError: If Cholesky decomposition fails

        Note:
            - The method doesn't guarantee exact correlation matching due to
              statistical variability and finite sample effects
            - For h_control=True, DFA analysis adds computational overhead
            - Large datasets benefit from parallel processing (threads > 1)
        """

        result = pd.DataFrame()

        if not isinstance(corr_target, Iterable):
            self.corr_target = [corr_target]
        else:
            self.corr_target = corr_target

        for corr in self.corr_target:
            indices = [corr for i in range(self.quantity)]

            if isinstance(self.corr_target, ndarray):
                Rxx = self.corr_target
            else:
                Rxx = full((self.quantity, self.quantity), corr, dtype=float)
                for i, j in zip(range(self.quantity), range(self.quantity)):
                    Rxx[i][j] = 1

            chol_transform_matrix = cholesky(Rxx)
            correlated_vectors = pd.DataFrame(
                matmul(chol_transform_matrix, self.dataset), index=indices
            )

            if h_control:
                if self.threads != 1:
                    h_estimated = DFA(correlated_vectors.to_numpy()).parallel_2d(
                        threads=self.threads
                    )
                else:
                    h_estimated = DFA(correlated_vectors.to_numpy()).find_h()

                correlated_vectors.insert(0, "H_est", h_estimated)

            if isinstance(self.corr_target, ndarray):
                return correlated_vectors

            result = (
                correlated_vectors
                if result.size == 0
                else result.append(correlated_vectors)
            )

        return result


if __name__ == "__main__":
    "Simple test. Here I create (100, 1024) array with given H = 1.5 then transform it" "using Cholesky distribution"

    d = FilteredArray(1.5, 1024, set_mean=10, set_std=3).generate(n_vectors=100)

    x = CorrelatedArray(data=d, threads=1).create(0.7, h_control=True).to_numpy()
