import numpy as np
import os


def load_col_major(path):
    mat = np.load(path)
    n, m = mat.shape
    mat = mat.reshape((m, n))
    mat = mat.transpose()

    return mat


def check_isclose(A, B):
    if (A.shape == B.shape):
        is_close = np.isclose(A, B)
        ratio_equal = is_close.sum() / B.size
    else:
        print(A.shape)
        print(B.shape)
        return 0

    return ratio_equal


def test_transpose():
    home = os.getenv("HOME")
    dir_path = home + "/gpu_memory_reduction/alzheimer/data"
    flickr_dir_path = dir_path + "/flickr"
    test_dir_path = dir_path + "/tests"

    path = test_dir_path + "/matrix.npy"
    matrix = np.load(path)
    n, m = matrix.shape
    matrix = matrix.reshape((m, n))
    path = test_dir_path + "/matrix_transposed.npy"
    matrix_transposed = np.load(path)
    path = test_dir_path + "/matrix_transposed_inplace.npy"
    matrix_transposed_inplace = np.load(path)

    ratio_equal = check_isclose(matrix.T, matrix_transposed)
    print("Matrix, matrix transposed: Ratio equal {}".format(ratio_equal))
    ratio_equal = check_isclose(matrix.T, matrix_transposed_inplace)
    print("Matrix, matrix transposed inplace: Ratio equal {}".format(ratio_equal))


if __name__ == "__main__":
    test_transpose()
