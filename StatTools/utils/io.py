from pathlib import Path

import numpy as np
from sympy import Matrix, sympify


def save_np_matrix(matrix: np.ndarray, file_path: Path):
    """Save a matrix to a file (path will be created if not exists)."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(file_path, matrix)


def load_np_matrix(file_path: str) -> np.ndarray:
    """Load a matrix from a file."""
    return np.load(file_path)


def save_sympy_matrix(matrix: Matrix, file_path: Path):
    """Save a matrix to a file (path will be created if not exists)."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(str(matrix.tolist()))


def load_sympy_matrix(file_path: str) -> Matrix:
    """Load a matrix from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    matrix_list = sympify(content)
    return Matrix(matrix_list)