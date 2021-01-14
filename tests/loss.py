import numpy as np
import os
import torch
from helper import (print_close_equal, check_close_equal,
        to_torch, write_return, update_return)


def test_loss():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_close = True
    home = os.getenv("HOME")
    dir_path = home + "/gpu_memory_reduction/alzheimer/data"
    flickr_dir_path = dir_path + "/flickr"
    test_dir_path = dir_path + "/tests"

    path = flickr_dir_path + "/classes.npy"
    classes = np.load(path)
    path = test_dir_path + "/input.npy"
    input_ = np.load(path)
    path = test_dir_path + "/loss.npy"
    loss = np.load(path)
    loss = loss.squeeze()
    path = test_dir_path + "/gradients.npy"
    gradients = np.load(path)

    # forward
    input_torch = to_torch(input_)
    classes_torch = torch.from_numpy(classes)
    classes_torch = classes_torch.long().to(device)
    loss_layer = torch.nn.NLLLoss()
    loss_torch = loss_layer(input_torch, classes_torch)
    true_loss = loss_torch.detach().cpu().numpy()
    
    ratio_close, ratio_equal = check_close_equal(loss, true_loss)
    all_close = update_return(ratio_close)
    print_close_equal("Loss", ratio_close, ratio_equal)
    
    # backward
    loss_torch.backward()
    true_gradients = input_torch.grad.detach().cpu().numpy()

    ratio_close, ratio_equal = check_close_equal(gradients, true_gradients)
    all_close = update_return(ratio_close)
    print_close_equal("Loss gradients", ratio_close, ratio_equal)

    path = test_dir_path + "/value.npy"
    write_return(all_close, path)


if __name__ == "__main__":
    test_loss()

