import numpy as np
import os
import torch
from helper import (print_close_equal, check_close_equal,
        to_torch, write_return, update_return, count_nans)


def test_log_softmax():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_close = True
    home = os.getenv("HOME")
    flickr_dir_path = "/mnt/data/flickr"
    test_dir_path = home + "/gpu_memory_reduction/alzheimer/data/tests"

    path = flickr_dir_path + "/features.npy"
    features = np.load(path)
    path = test_dir_path + "/output.npy"
    activations = np.load(path)
    path = test_dir_path + "/incoming_gradients.npy"
    in_gradients = np.load(path)
    path = test_dir_path + "/gradients.npy"
    gradients = np.load(path)

    # forward
    features_torch = to_torch(features)
    log_softmax_layer = torch.nn.LogSoftmax(dim=-1)
    activations_torch = log_softmax_layer(features_torch)
    true_activations = activations_torch.detach().cpu().numpy()

    ratio_close, ratio_equal = check_close_equal(activations, true_activations)
    all_close = update_return(ratio_close) 
    print_close_equal("LogSoftmax", ratio_close, ratio_equal)

    num_nans = count_nans(activations)
    if (num_nans > 0):
        print("LogSoftmax has {} NaNs".format(num_nans))
        path = test_dir_path + "/value.npy"
        write_return(0.0, path)

    # backward
    in_gradients_torch = torch.from_numpy(in_gradients)
    in_gradients_torch = in_gradients_torch.to(device)
    activations_torch.backward(in_gradients_torch)
    true_gradients = features_torch.grad.detach().cpu().numpy()

    ratio_close, ratio_equal = check_close_equal(gradients, true_gradients)
    all_close = update_return(ratio_close) 
    print_close_equal("LogSoftmax gradients", ratio_close, ratio_equal)

    num_nans = count_nans(gradients)
    if (num_nans > 0):
        print("LogSoftmax gradients has {} NaNs".format(num_nans))
        path = test_dir_path + "/value.npy"
        write_return(0.0, path)

    path = test_dir_path + "/value.npy"
    write_return(all_close, path)


if __name__ == "__main__":
    test_log_softmax()

