import numpy as np
import os
from helper import (check_close_equal, write_equal, print_close_equal,
        load_col_major)

def reshape_mat(mat):
    n, m = mat.shape
    return mat.reshape(m, n)

def test_transpose():
    home = os.getenv("HOME")
    dir_path = home + "/gpu_memory_reduction/alzheimer/data"
    flickr_dir_path = dir_path + "/flickr"
    test_dir_path = dir_path + "/tests"

    path = test_dir_path + "/to_col_major.npy"
    to_col_major_np = np.load(path)
    to_col_major = to_col_major_np[0].item()

    path_mat = test_dir_path + "/matrix.npy"
    path_mat_T = test_dir_path + "/matrix_transposed.npy"
    if (to_col_major == 1):
        matrix = np.load(path_mat)
        matrix_transposed = np.load(path_mat_T)
        matrix_transposed = reshape_mat(matrix_transposed)
    else:
        matrix = np.load(path_mat)
        matrix = reshape_mat(matrix)
        matrix_transposed = np.load(path_mat_T)

    ratio_close, ratio_equal = check_close_equal(matrix.T, matrix_transposed)
    print_close_equal("Matrix, matrix transpose", ratio_close, ratio_equal)

    path = test_dir_path + "/value.npy"
    write_equal(ratio_equal, path)


if __name__ == "__main__":
    test_transpose()
