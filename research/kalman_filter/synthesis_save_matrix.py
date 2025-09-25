import multiprocessing
import warnings
from pathlib import Path

import numpy as np
from tqdm import tqdm

from StatTools.filters.symbolic_kalman import (
    get_sympy_filter_matrix,
    refine_filter_matrix,
)
from StatTools.generators.kasdin_generator import create_kasdin_generator
from StatTools.utils import io

warnings.filterwarnings("ignore")


def process_single_iter(args):
    """
    Process one iteration with H and order.

    Args:
        h (float): Hurst exponent.
        r (int): filter order.
    """
    h, r = args
    file_path = Path("filter_matrices") / f"{h}_{r}.npy"
    if file_path.is_file():
        return file_path
    length = r + 1

    generator = create_kasdin_generator(h, length=length)
    ar_filter = generator.get_filter_coefficients()
    if r == 1:
        # Simple position-velocity model
        io.save_matrix(np.array([[1]]), file_path)
        return file_path
    number_matrix = refine_filter_matrix(get_sympy_filter_matrix(r), r, ar_filter)
    io.save_matrix(np.array(number_matrix, dtype=np.float64), file_path)
    return file_path


if __name__ == "__main__":
    H_LIST = np.arange(0.5, 3.75, 0.25)
    R_LIST = np.array([2**i for i in range(1, 8)])
    print(H_LIST, R_LIST)
    args_list = []
    for h in H_LIST:
        for r in R_LIST:
            args_list.append((h, r))
    print(f"Got {len(args_list)} combinations.")

    print("Run.")
    # with multiprocessing.Pool() as pool:
    #     results = list(
    #         tqdm(
    #             pool.imap_unordered(process_single_iter, args_list),
    #             total=len(args_list),
    #             desc="Progress",
    #         )
    #     )
    for arg in tqdm(args_list):
        process_single_iter(arg)
        
    print("Done.")
