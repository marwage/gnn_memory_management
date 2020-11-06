import numpy as np
import os
import scipy.io
import scipy.sparse as sp
import torch


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


def print_nan_coor(A):
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if np.isnan(A[i, j]):
                print("NaN at ({}, {})".format(i, j))

def num_close_rows(A, B):
    is_close = np.isclose(A, B)
    is_close_sum = is_close.sum(axis=1)
    close_rows = is_close_sum == A.shape[1]
    
    return close_rows.sum()


def test_adam():
    home = os.getenv("HOME")
    dir_path = home + "/gpu_memory_reduction/alzheimer/data"
    flickr_dir_path = dir_path + "/flickr"
    test_dir_path = dir_path + "/tests"

    for i in range(4):
        path = test_dir_path + "/gradient_" + str(i) + ".npy"
        gradient = load_col_major(path)

        path = test_dir_path + "/gradient_chunked_" + str(i) + ".npy"
        gradient_chunked = load_col_major(path)

        ratio = check_isclose(gradient, gradient_chunked)
        print("Gradient of layer {}: Ratio {}".format(i, ratio))

        path = test_dir_path + "/weight_" + str(i) + ".npy"
        weight = load_col_major(path)

        path = test_dir_path + "/weight_chunked_" + str(i) + ".npy"
        weight_chunked = load_col_major(path)

        ratio = check_isclose(weight, weight_chunked)
        print("Weight of layer {}: Ratio {}".format(i, ratio))

        # debug
        print("gradients")
        print(gradient[0:5, 0:5])
        print(gradient_chunked[0:5, 0:5])
        print("weights")
        print(weight[0:5, 0:5])
        print(weight_chunked[0:5, 0:5])


if __name__ == "__main__":
    test_adam()

