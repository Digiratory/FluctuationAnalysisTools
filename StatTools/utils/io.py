from pathlib import Path

import numpy as np


def save_matrix(matrix: np.ndarray, file_path: Path):
    """Save a matrix to a file (path will be created if not exists)."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(file_path, matrix)


def load_matrix(file_path: str) -> np.ndarray:
    """Load a matrix from a file."""
    return np.load(file_path)
