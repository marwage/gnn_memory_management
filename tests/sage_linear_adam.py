import numpy as np
import os
import scipy.io
import scipy.sparse as sp
import torch
from helper import (write_equal, print_close_equal, to_torch,
                    check_close_equal)


def test_sage_linear_adam():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_close = 1.0

    home = os.getenv("HOME")
    dir_path = home + "/gpu_memory_reduction/alzheimer/data"
    flickr_dir_path = dir_path + "/flickr"
    test_dir_path = dir_path + "/tests"

    # FORWARD
    path = test_dir_path + "/input_self.npy"
    input_self = np.load(path)
    path = test_dir_path + "/input_neigh.npy"
    input_neigh = np.load(path)
    path = test_dir_path + "/in_gradients.npy"
    in_gradients = np.load(path)

    input_self_torch = to_torch(input_self)
    input_neigh_torch = to_torch(input_neigh)

    path = test_dir_path + "/self_weight.npy"
    self_weight = np.load(path)
    path = test_dir_path + "/self_bias.npy"
    self_bias = np.load(path)
    path = test_dir_path + "/neigh_weight.npy"
    neigh_weight = np.load(path)
    path = test_dir_path + "/neigh_bias.npy"
    neigh_bias = np.load(path)
    path = test_dir_path + "/activations.npy"
    activations = np.load(path)

    self_weight_torch = to_torch(self_weight)
    self_bias_torch = to_torch(self_bias)
    neigh_weight_torch = to_torch(neigh_weight)
    neigh_bias_torch = to_torch(neigh_bias)

    learning_rate = 0.003
    params = [self_weight_torch,
              self_bias_torch,
              neigh_weight_torch,
              neigh_bias_torch]
    optimiser = torch.optim.Adam(params, lr=learning_rate)
    optimiser.zero_grad()

    self_result_torch = torch.matmul(input_self_torch, self_weight_torch) + self_bias_torch.T
    neigh_result_torch = torch.matmul(input_neigh_torch, neigh_weight_torch) + neigh_bias_torch.T
    true_sage_linear_result_torch = self_result_torch + neigh_result_torch

    true_sage_linear_result = true_sage_linear_result_torch.detach().cpu().numpy()

    ratio_close, ratio_equal = check_close_equal(activations, true_sage_linear_result)
    all_close = all_close * ratio_close
    print_close_equal("SageLinear", ratio_close, ratio_equal)

    # BACKPROPAGATION
    in_gradients_torch = torch.from_numpy(in_gradients)
    in_gradients_torch = in_gradients_torch.to(device)
    true_sage_linear_result_torch.backward(in_gradients_torch)

    path = test_dir_path + "/self_grads.npy"
    self_grads = np.load(path)
    path = test_dir_path + "/neigh_grads.npy"
    neigh_grads = np.load(path)

    path = test_dir_path + "/self_weight_grads.npy"
    self_weight_grads = np.load(path)
    path = test_dir_path + "/self_bias_grads.npy"
    self_bias_grads = np.load(path)
    path = test_dir_path + "/neigh_weight_grads.npy"
    neigh_weight_grads = np.load(path)
    path = test_dir_path + "/neigh_bias_grads.npy"
    neigh_bias_grads = np.load(path)

    true_self_grads = input_self_torch.grad.cpu().numpy()
    true_neigh_grads = input_neigh_torch.grad.cpu().numpy()
    true_self_weight_grads = self_weight_torch.grad.cpu().numpy()
    true_self_bias_grads = self_bias_torch.grad.cpu().numpy()
    true_neigh_weight_grads = neigh_weight_torch.grad.cpu().numpy()
    true_neigh_bias_grads = neigh_bias_torch.grad.cpu().numpy()

    ratio_close, ratio_equal = check_close_equal(self_grads, true_self_grads)
    all_close = all_close * ratio_close
    print_close_equal("LinearSage self", ratio_close, ratio_equal)
    ratio_close, ratio_equal = check_close_equal(neigh_grads, true_neigh_grads)
    all_close = all_close * ratio_close
    print_close_equal("LinearSage neighbourhood", ratio_close, ratio_equal)
    ratio_close, ratio_equal = check_close_equal(self_weight_grads, true_self_weight_grads)
    all_close = all_close * ratio_close
    print_close_equal("LinearSage self weight", ratio_close, ratio_equal)
    ratio_close, ratio_equal = check_close_equal(self_bias_grads, true_self_bias_grads)
    all_close = all_close * ratio_close
    print_close_equal("LinearSage self bias", ratio_close, ratio_equal)
    ratio_close, ratio_equal = check_close_equal(neigh_weight_grads, true_neigh_weight_grads)
    all_close = all_close * ratio_close
    print_close_equal("LinearSage neighbourhood weight", ratio_close, ratio_equal)
    ratio_close, ratio_equal = check_close_equal(neigh_bias_grads, true_neigh_bias_grads)
    all_close = all_close * ratio_close
    print_close_equal("LinearSage neighbourhood bias", ratio_close, ratio_equal)

    # ADAM
    optimiser.step()

    path = test_dir_path + "/self_weight_updated.npy";
    self_weight_updated = np.load(path)
    path = test_dir_path + "/self_bias_updated.npy";
    self_bias_updated = np.load(path)
    path = test_dir_path + "/neigh_weight_updated.npy";
    neigh_weight_updated = np.load(path)
    path = test_dir_path + "/neigh_bias_updated.npy";
    neigh_bias_updated = np.load(path)

    true_self_weight_updated = self_weight_torch.detach().cpu().numpy()
    ratio_close, ratio_equal = check_close_equal(self_weight_updated, true_self_weight_updated)
    all_close = all_close * ratio_close
    print_close_equal("Adam self weight", ratio_close, ratio_equal)

    true_self_bias_updated = self_bias_torch.detach().cpu().numpy()
    ratio_close, ratio_equal = check_close_equal(self_bias_updated, true_self_bias_updated)
    all_close = all_close * ratio_close
    print_close_equal("Adam self bias", ratio_close, ratio_equal)

    true_neigh_weight_updated = neigh_weight_torch.detach().cpu().numpy()
    ratio_close, ratio_equal = check_close_equal(neigh_weight_updated, true_neigh_weight_updated)
    all_close = all_close * ratio_close
    print_close_equal("Adam neighbourhood weight", ratio_close, ratio_equal)

    true_neigh_bias_updated = neigh_bias_torch.detach().cpu().numpy()
    ratio_close, ratio_equal = check_close_equal(neigh_bias_updated, true_neigh_bias_updated)
    all_close = all_close * ratio_close
    print_close_equal("Adam neighbourhood bias", ratio_close, ratio_equal)

    path = test_dir_path + "/value.npy"
    write_equal(all_close, path)


def compare_adam():
    home = os.getenv("HOME")
    dir_path = home + "/gpu_memory_reduction/alzheimer/data"
    flickr_dir_path = dir_path + "/flickr"
    test_dir_path = dir_path + "/tests"
    all_equal = 1.0

    for i in range(4):
        path = test_dir_path + "/gradient_" + str(i) + ".npy"
        gradient = np.load(path)

        path = test_dir_path + "/gradient_chunked_" + str(i) + ".npy"
        gradient_chunked = np.load(path)

        ratio_close, ratio_equal = check_close_equal(gradient, gradient_chunked)
        all_equal = all_equal * ratio_equal
        print_close_equal("Gradient layer {}".format(i), ratio_close, ratio_equal)

        path = test_dir_path + "/weight_" + str(i) + ".npy"
        weight = np.load(path)

        path = test_dir_path + "/weight_chunked_" + str(i) + ".npy"
        weight_chunked = np.load(path)

        ratio_close, ratio_equal = check_close_equal(weight, weight_chunked)
        all_equal = all_equal * ratio_equal
        print_close_equal("Weight layer {}".format(i), ratio_close, ratio_equal)

    path = test_dir_path + "/value.npy"
    write_equal(all_equal, path)


if __name__ == "__main__":
    test_sage_linear_adam()
