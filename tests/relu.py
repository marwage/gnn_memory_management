import numpy as np
import os
import torch
from helper import (print_close_equal, check_close_equal,
        to_torch, print_not_close, write_return, update_return)


def test_relu():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_close = True
    home = os.getenv("HOME")
    dir_path = home + "/gpu_memory_reduction/alzheimer/data"
    flickr_dir_path = dir_path + "/flickr"
    test_dir_path = dir_path + "/tests"

    path = flickr_dir_path + "/features.npy"
    features = np.load(path)
    path = test_dir_path + "/activations.npy"
    activations = np.load(path)
    path = test_dir_path + "/gradients.npy"
    gradients = np.load(path)
    path = test_dir_path + "/in_gradients.npy"
    in_gradients = np.load(path)

    relu_layer = torch.nn.ReLU()

    # forward
    features_torch = to_torch(features)
    activations_torch = relu_layer(features_torch)
    true_activations = activations_torch.detach().cpu().numpy()

    ratio_close, ratio_equal = check_close_equal(activations, true_activations)
    all_close = update_return(ratio_close)
    print_close_equal("ReLU", ratio_close, ratio_equal)

    # backward
    in_gradients_torch = torch.from_numpy(in_gradients)
    in_gradients_torch = in_gradients_torch.to(device)
    activations_torch.backward(in_gradients_torch)
    true_gradients = features_torch.grad.detach().cpu().numpy()

    ratio_close, ratio_equal = check_close_equal(gradients, true_gradients)
    all_close = update_return(ratio_close)
    print_close_equal("ReLU gradients", ratio_close, ratio_equal)

    path = test_dir_path + "/value.npy"
    write_return(all_close, path)


if __name__ == "__main__":
    test_relu()

