import numpy as np
import os
from helper import (check_close_equal, write_equal, print_close_equal)


def test_transpose():
    home = os.getenv("HOME")
    dir_path = home + "/gpu_memory_reduction/alzheimer/data"
    flickr_dir_path = dir_path + "/flickr"
    test_dir_path = dir_path + "/tests"

    path = test_dir_path + "/matrix.npy"
    matrix = np.load(path)
    path = test_dir_path + "/matrix_transposed.npy"
    matrix_transposed = np.load(path)
    n, m = matrix_transposed.shape
    matrix_transposed = matrix_transposed.reshape(m, n)

    ratio_close, ratio_equal = check_close_equal(matrix.T, matrix_transposed)
    print_close_equal("Matrix, matrix transpose", ratio_close, ratio_equal)

    path = test_dir_path + "/value.npy"
    write_equal(ratio_equal, path)


if __name__ == "__main__":
    test_transpose()
