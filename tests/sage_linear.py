import numpy as np
import os
import scipy.io
import scipy.sparse as sp
import torch
from helper import (check_equal, check_isclose, save_return_value,
                    write_equal, check_nans, print_small, print_close_equal)


def test_sage_linear():
    all_close = 1.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    home = os.getenv("HOME")
    dir_path = home + "/gpu_memory_reduction/alzheimer/data"
    flickr_dir_path = dir_path + "/flickr"
    test_dir_path = dir_path + "/tests"

    path = test_dir_path + "/input_self.npy"
    input_self = np.load(path)
    path = test_dir_path + "/input_neigh.npy"
    input_neigh = np.load(path)
    path = test_dir_path + "/in_gradients.npy"
    in_gradients = np.load(path)

    check_nans(input_self, "Input for self")
    check_nans(input_neigh, "Input for neighbourhood")
    check_nans(in_gradients, "Incoming gradients")

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
    self_weight = np.load(path)
    path = test_dir_path + "/self_bias.npy"
    self_bias = np.load(path)
    path = test_dir_path + "/neigh_weight.npy"
    neigh_weight = np.load(path)
    path = test_dir_path + "/neigh_bias.npy"
    neigh_bias = np.load(path)
    path = test_dir_path + "/result.npy"
    sage_linear_result = np.load(path)

    check_nans(self_weight, "Self weight")
    check_nans(self_bias, "Self bias")
    check_nans(neigh_weight, "Neighbourhood weight")
    check_nans(neigh_bias, "Neighbourhood bias")
    check_nans(sage_linear_result, "SageLinear forward")

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

    ratio_close = check_isclose(sage_linear_result, true_sage_linear_result)
    ratio_equal = check_equal(sage_linear_result, true_sage_linear_result)
    print_close_equal("SageLinear forward", ratio_close, ratio_equal)
    all_close = all_close * ratio_close

    # BACKPROPAGATION
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

    ratio_close = check_isclose(self_grads, true_self_grads)
    ratio_equal = check_equal(self_grads, true_self_grads)
    print("Linear self input: Close: {}, Equal: {}".format(ratio_close, ratio_equal))
    all_close = all_close * ratio_close

    ratio_close = check_isclose(neigh_grads, true_neigh_grads)
    ratio_equal = check_equal(neigh_grads, true_neigh_grads)
    print("Linear neigh input: Close: {}, Equal: {}".format(ratio_close, ratio_equal))
    all_close = all_close * ratio_close

    ratio_close = check_isclose(self_weight_grads, true_self_weight_grads)
    ratio_equal = check_equal(self_weight_grads, true_self_weight_grads)
    print("Linear self weight: Close: {}, Equal: {}".format(ratio_close, ratio_equal))
    all_close = all_close * ratio_close

    ratio_close = check_isclose(self_weight_grads, true_self_weight_grads, True)
    print("Linear self weight: Loose: {}".format(ratio_close))

    ratio_close = check_isclose(self_bias_grads, true_self_bias_grads)
    ratio_equal = check_equal(self_bias_grads, true_self_bias_grads)
    print("Linear self bias: Close: {}, Equal: {}".format(ratio_close, ratio_equal))
    all_close = all_close * ratio_close

    ratio_close = check_isclose(self_bias_grads, true_self_bias_grads, True)
    print("Linear self bias: Loose {}".format(ratio_close))

    ratio_close = check_isclose(neigh_weight_grads, true_neigh_weight_grads)
    ratio_equal = check_equal(neigh_weight_grads, true_neigh_weight_grads)
    print("Linear neigh weight: Close: {}, Equal: {}".format(ratio_close, ratio_equal))
    all_close = all_close * ratio_close

    ratio_close = check_isclose(neigh_weight_grads, true_neigh_weight_grads, True)
    print("Linear neigh weight: Loose: {}".format(ratio_close))

    ratio_close = check_isclose(neigh_bias_grads, true_neigh_bias_grads)
    ratio_equal = check_equal(neigh_bias_grads, true_neigh_bias_grads)
    print("Linear neigh bias: Close: {}, Equal: {}".format(ratio_close, ratio_equal))
    all_close = all_close * ratio_close

    ratio_close = check_isclose(neigh_bias_grads, true_neigh_bias_grads, True)
    print("Linear neigh bias: Loose: {}".format(ratio_close))

    path = test_dir_path + "/value.npy"
    write_equal(all_close, path)


if __name__ == "__main__":
    test_sage_linear()
