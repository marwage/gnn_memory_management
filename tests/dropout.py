import numpy as np
import os
import torch
from helper import (check_close_equal, print_close_equal, write_equal, count_nans)


def test_dropout():
    all_equal = 1.0
    home = os.getenv("HOME")
    test_dir_path = home + "/gpu_memory_reduction/alzheimer/data/tests"
    flickr_dir_path = "/mnt/data/flickr"

    path = flickr_dir_path + "/features.npy"
    features = np.load(path)
    path = test_dir_path + "/output.npy"
    dropout_result = np.load(path)

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
    num_nans = count_nans(dropout_result)
    if (num_nans > 0):
        print("Dropout has {} NaNs".format(num_nans))

    path = test_dir_path + "/gradients.npy"
    dropout_grads = np.load(path)
    path = test_dir_path + "/incoming_gradients.npy"
    in_gradients = np.load(path)

    in_gradients_torch = torch.from_numpy(in_gradients)
    dropout_result_torch.backward(in_gradients_torch)
    true_dropout_grads = features_torch.grad.numpy()

    ratio_close, ratio_equal = check_close_equal(dropout_grads, true_dropout_grads)
    all_equal = all_equal * ratio_equal
    print_close_equal("Dropout gradients", ratio_close, ratio_equal)
    num_nans = count_nans(dropout_grads)
    if (num_nans > 0):
        print("Dropout gradients has {} NaNs".format(num_nans))

    path = test_dir_path + "/value.npy"
    write_equal(all_equal, path)


if __name__ == "__main__":
    test_dropout()

