import numpy as np
import pytest
from sympy import Matrix

from StatTools.utils import io


def test_save_load_np_matrix(tmp_path):
    mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    file_path = tmp_path / "test.npy"
    io.save_np_matrix(mat, file_path)
    mat_est = io.load_np_matrix(file_path)
    assert np.allclose(mat, mat_est)


def test_save_load_sympy_matrix(tmp_path):
    mat = Matrix([[1, -1], [3, 4], [0, 2]])
    file_path = tmp_path / "test.txt"
    io.save_sympy_matrix(mat, file_path)
    mat_est = io.load_sympy_matrix(file_path)
    assert mat.equals(mat_est)
