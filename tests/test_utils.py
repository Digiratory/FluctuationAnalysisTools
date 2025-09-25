import numpy as np
import pytest

from StatTools.utils import io


def test_save_load_matrix(tmp_path):
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    file_path = tmp_path / "test.npy"
    io.save_matrix(arr, file_path)
    arr_est = io.load_matrix(file_path)
    assert np.allclose(arr, arr_est)
