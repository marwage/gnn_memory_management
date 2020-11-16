import numpy as np
import os
import torch
from helper import (load_col_major, check_close_equal, print_close_equal, write_equal)


def test_dropout():
    all_equal = 1.0
    home = os.getenv("HOME")
    dir_path = home + "/gpu_memory_reduction/alzheimer/data"
    flickr_dir_path = dir_path + "/flickr"
    test_dir_path = dir_path + "/tests"

    path = flickr_dir_path + "/features.npy"
    features = np.load(path)
    path = test_dir_path + "/dropout_result.npy"
    dropout_result = load_col_major(path)

    probability = 0.2
    dropout_layer = torch.nn.Dropout(p=probability)

    features_torch = torch.from_numpy(features)
    features_torch.requires_grad_()
    features_torch.retain_grad()
    dropout_result_torch = dropout_layer(features_torch)
    true_dropout_result = dropout_result_torch.detach().numpy()  # Not really true due to randomness

    ratio_close, ratio_equal = check_close_equal(dropout_result, true_dropout_result)
    all_equal = all_equal * ratio_equal
    print_close_equal("Dropout", ratio_close, ratio_equal)

    path = test_dir_path + "/dropout_gradients.npy"
    dropout_grads = load_col_major(path)
    path = test_dir_path + "/in_gradients.npy"
    in_gradients = load_col_major(path)

    in_gradients_torch = torch.from_numpy(in_gradients)
    dropout_result_torch.backward(in_gradients_torch)
    true_dropout_grads = features_torch.grad.numpy()

    ratio_close, ratio_equal = check_close_equal(dropout_grads, true_dropout_grads)
    all_equal = all_equal * ratio_equal
    print_close_equal("Dropout gradients", ratio_close, ratio_equal)

    path = test_dir_path + "/value.npy"
    write_equal(all_equal, path)


if __name__ == "__main__":
    test_dropout()
