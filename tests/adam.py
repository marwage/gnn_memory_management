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

def to_torch(a):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    a_torch = torch.from_numpy(a)
    a_torch = a_torch.to(device)
    a_torch.requires_grad_()
    a_torch.retain_grad()

    return a_torch


def test_adam():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    home = os.getenv("HOME")
    dir_path = home + "/gpu_memory_reduction/alzheimer/data"
    flickr_dir_path = dir_path + "/flickr"
    test_dir_path = dir_path + "/tests"

    path = test_dir_path + "/weight.npy"
    weight = load_col_major(path)
    path = test_dir_path + "/bias.npy"
    bias = load_col_major(path)
    path = test_dir_path + "/weight_grads.npy"
    weight_grads = load_col_major(path)
    path = test_dir_path + "/bias_grads.npy"
    bias_grads = load_col_major(path)

    weight_torch = to_torch(weight)
    bias_torch = to_torch(bias)
    weight_grads_torch = torch.from_numpy(weight_grads)
    weight_grads_torch = weight_grads_torch.to(device)
    bias_grads_torch = torch.from_numpy(bias_grads)
    bias_grads_torch = bias_grads_torch.to(device)

    learning_rate = 0.003
    params = [weight_torch, bias_torch]
    optimiser = torch.optim.Adam(params, lr=learning_rate)
    optimiser.zero_grad()

    weight_torch.grad = weight_grads_torch
    bias_torch.grad = bias_grads_torch

    optimiser.step()

    path = test_dir_path + "/weight_updated.npy";
    weight_updated = load_col_major(path)
    path = test_dir_path + "/bias_updated.npy";
    bias_updated = load_col_major(path)

    true_weight = weight_torch.detach().cpu().numpy()
    ratio_equal = check_isclose(weight_updated, true_weight)
    print("Adam weight: Ratio of equal elements {}".format(ratio_equal))

    true_bias = bias_torch.detach().cpu().numpy()
    ratio_equal = check_isclose(bias_updated, true_bias)
    print("Adam bias: Ratio of equal elements {}".format(ratio_equal))
    #  print_small(bias_updated)
    #  print_small(true_bias)


if __name__ == "__main__":
    test_adam()

