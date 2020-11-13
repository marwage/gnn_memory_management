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

def print_small(A):
    print(A[0:3, 0:3])

def breakpoint():
    import os, signal
    os.kill(os.getpid(), signal.SIGTRAP)

def print_unequal(A, B):
    is_close = np.isclose(A, B)
    num_rows, num_columns = A.shape
    for i in range(num_rows):
        for j in range(num_columns):
            if not is_close[i, j]:
                print("Coordinate: ({}, {})".format(i, j))
                print("Values: {}, {}; Diff: {}".format(A[i, j], B[i, j], A[i, j] - B[i, j]))


def test_sage_linear():
    product_equals = 1.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    home = os.getenv("HOME")
    dir_path = home + "/gpu_memory_reduction/alzheimer/data"
    flickr_dir_path = dir_path + "/flickr"
    test_dir_path = dir_path + "/tests"

    path = test_dir_path + "/input_self.npy"
    input_self = load_col_major(path)
    path = test_dir_path + "/input_neigh.npy"
    input_neigh = load_col_major(path)
    path = test_dir_path + "/in_gradients.npy"
    in_gradients = load_col_major(path)

    input_self_torch = torch.from_numpy(input_self)
    input_self_torch = input_self_torch.to(device)
    input_self_torch.requires_grad_()
    input_self_torch.retain_grad()
    input_neigh_torch = torch.from_numpy(input_neigh)
    input_neigh_torch = input_neigh_torch.to(device)
    input_neigh_torch.requires_grad_()
    input_neigh_torch.retain_grad()
    in_gradients_torch = torch.from_numpy(in_gradients)
    in_gradients_torch = in_gradients_torch.to(device)

    path = test_dir_path + "/self_weight.npy"
    self_weight = load_col_major(path)
    path = test_dir_path + "/self_bias.npy"
    self_bias = load_col_major(path)
    path = test_dir_path + "/neigh_weight.npy"
    neigh_weight = load_col_major(path)
    path = test_dir_path + "/neigh_bias.npy"
    neigh_bias = load_col_major(path)
    path = test_dir_path + "/result.npy"
    sage_linear_result = load_col_major(path)
    #  sage_linear_result = np.load(path) # DEBUGGING result is row major

    self_weight_torch = torch.from_numpy(self_weight)
    self_weight_torch = self_weight_torch.to(device)
    self_weight_torch.requires_grad_()
    self_weight_torch.retain_grad()
    self_bias_torch = torch.from_numpy(self_bias)
    self_bias_torch = self_bias_torch.to(device)
    self_bias_torch.requires_grad_()
    self_bias_torch.retain_grad()

    neigh_weight_torch = torch.from_numpy(neigh_weight)
    neigh_weight_torch = neigh_weight_torch.to(device)
    neigh_weight_torch.requires_grad_()
    neigh_weight_torch.retain_grad()
    neigh_bias_torch = torch.from_numpy(neigh_bias)
    neigh_bias_torch = neigh_bias_torch.to(device)
    neigh_bias_torch.requires_grad_()
    neigh_bias_torch.retain_grad()

    self_result_torch = torch.matmul(input_self_torch, self_weight_torch) + self_bias_torch.T
    neigh_result_torch = torch.matmul(input_neigh_torch, neigh_weight_torch) + neigh_bias_torch.T
    true_sage_linear_result_torch = self_result_torch + neigh_result_torch

    true_sage_linear_result = true_sage_linear_result_torch.detach().cpu().numpy()

    ratio_equal = check_isclose(sage_linear_result, true_sage_linear_result)
    print("SageLinear: Percentage of equal elements: {}".format(ratio_equal))
    #  print_unequal(sage_linear_result, true_sage_linear_result)

    if (ratio_equal == 1.0):
        value = np.array([1], dtype=np.int32)
    else:
        value = np.array([0], dtype=np.int32)
    path = test_dir_path + "/value.npy"
    np.save(path, value)


if __name__ == "__main__":
    test_sage_linear()

