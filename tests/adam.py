import numpy as np
import os
import scipy.io
import scipy.sparse as sp
import torch
from helper import (check_isclose, write_equal,
                    to_torch, check_equal, print_close_equal)


def test_adam():
    all_equal = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    home = os.getenv("HOME")
    dir_path = home + "/gpu_memory_reduction/alzheimer/data"
    flickr_dir_path = dir_path + "/flickr"
    test_dir_path = dir_path + "/tests"

    path = test_dir_path + "/weight.npy"
    weight = np.load(path)
    path = test_dir_path + "/bias.npy"
    bias = np.load(path)
    path = test_dir_path + "/weight_grads.npy"
    weight_grads = np.load(path)
    path = test_dir_path + "/bias_grads.npy"
    bias_grads = np.load(path)

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
    weight_updated = np.load(path)
    path = test_dir_path + "/bias_updated.npy";
    bias_updated = np.load(path)

    true_weight = weight_torch.detach().cpu().numpy()
    ratio_close = check_isclose(weight_updated, true_weight)
    ratio_equal = check_equal(weight_updated, true_weight)
    all_equal = all_equal * ratio_equal
    print_close_equal("Adam weight", ratio_close, ratio_equal)

    true_bias = bias_torch.detach().cpu().numpy()
    ratio_close = check_isclose(bias_updated, true_bias)
    ratio_equal = check_equal(bias_updated, true_bias)
    all_equal = all_equal * ratio_equal
    print_close_equal("Adam bias", ratio_close, ratio_equal)

    path = test_dir_path + "/value.npy"
    write_equal(all_equal, path)


if __name__ == "__main__":
    test_adam()

