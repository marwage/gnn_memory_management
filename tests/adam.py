import numpy as np
import os
import scipy.io
import scipy.sparse as sp
import torch
from helper import load_col_major, check_isclose, print_nan_coor, num_close_rows, print_small, to_torch


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

