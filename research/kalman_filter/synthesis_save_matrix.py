import multiprocessing
import warnings
from pathlib import Path

import numpy as np
from tqdm import tqdm

from StatTools.experimental.filters.symbolic_kalman import (
    get_sympy_filter_matrix,
    refine_filter_matrix,
)
from StatTools.generators.kasdin_generator import create_kasdin_generator
from StatTools.utils import io

warnings.filterwarnings("ignore")


def process_r(args):
    """
    Process one iteration with H and order.

    Args:
        h (float): Hurst exponent.
        r (int): filter order.
    """
    h_list, r = args
    results = []
    matrix_file_path = Path("filter_matrices") / f"sympy_{r}.txt"
    if matrix_file_path.is_file():
        print(f"Loading sympy filter matrix r={r}")
        sympy_filter_matrix = io.load_sympy_matrix(matrix_file_path)
    else:
        print(f"Getting sympy filter matrix r={r}")
        sympy_filter_matrix = get_sympy_filter_matrix(r)
        io.save_sympy_matrix(sympy_filter_matrix, matrix_file_path)
    length = r + 1
    for h in h_list:
        file_path = Path("filter_matrices") / f"{h}_{r}.npy"
        results.append(file_path)
        if file_path.is_file():
            continue
        generator = create_kasdin_generator(h, length=length)
        ar_filter = generator.get_filter_coefficients()
        number_matrix = refine_filter_matrix(sympy_filter_matrix, r, ar_filter)
        io.save_np_matrix(np.array(number_matrix, dtype=np.float64), file_path)
    return results


if __name__ == "__main__":
    H_LIST = np.arange(0.5, 3.75, 0.25)
    R_LIST = np.array([2**i for i in range(1, 8)])
    print(H_LIST, R_LIST)
    args_list = []
    for r in R_LIST:
        args_list.append((H_LIST, r))

    print(f"Got {len(H_LIST) * len(R_LIST)} combinations.")

    print("Run.")
    with multiprocessing.Pool() as pool:
        results = list(
            tqdm(
                pool.imap_unordered(process_r, args_list),
                total=len(args_list),
                desc="Progress",
            )
        )

    print(f"Done. {len(results)} files.")
