import numpy as np
import os
import scipy.io
import scipy.sparse as sp
import torch
from helper import check_equal, check_isclose, load_col_major, save_return_value


def test_sage_linear():
    product_equals = 1.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    home = os.getenv("HOME")
    dir_path = home + "/gpu_memory_reduction/alzheimer/data"
    flickr_dir_path = dir_path + "/flickr"
    test_dir_path = dir_path + "/tests"

    path = test_dir_path + "/input_self.npy"
    input_self = load_col_major(path)
    path = test_dir_path + "/input_neigh.npy"
    input_neigh = load_col_major(path)
    path = test_dir_path + "/in_gradients.npy"
    in_gradients = load_col_major(path)

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
    self_weight = load_col_major(path)
    path = test_dir_path + "/self_bias.npy"
    self_bias = load_col_major(path)
    path = test_dir_path + "/neigh_weight.npy"
    neigh_weight = load_col_major(path)
    path = test_dir_path + "/neigh_bias.npy"
    neigh_bias = load_col_major(path)
    path = test_dir_path + "/result.npy"
    sage_linear_result = load_col_major(path)

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
    print("SageLinear: Close: {}, Equal: {}".format(ratio_close, ratio_equal))
    product_equals = product_equals * ratio_equal

    # BACKPROPAGATION
    true_sage_linear_result_torch.backward(in_gradients_torch)

    path = test_dir_path + "/self_grads.npy"
    self_grads = load_col_major(path)
    path = test_dir_path + "/neigh_grads.npy"
    neigh_grads = load_col_major(path)

    path = test_dir_path + "/self_weight_grads.npy"
    self_weight_grads = load_col_major(path)
    path = test_dir_path + "/self_bias_grads.npy"
    self_bias_grads = load_col_major(path)
    path = test_dir_path + "/neigh_weight_grads.npy"
    neigh_weight_grads = load_col_major(path)
    path = test_dir_path + "/neigh_bias_grads.npy"
    neigh_bias_grads = load_col_major(path)

    true_self_grads = input_self_torch.grad.cpu().numpy()
    true_neigh_grads = input_neigh_torch.grad.cpu().numpy()
    true_self_weight_grads = self_weight_torch.grad.cpu().numpy()
    true_self_bias_grads = self_bias_torch.grad.cpu().numpy()
    true_neigh_weight_grads = neigh_weight_torch.grad.cpu().numpy()
    true_neigh_bias_grads = neigh_bias_torch.grad.cpu().numpy()

    ratio_close = check_isclose(self_grads, true_self_grads)
    ratio_equal = check_equal(self_grads, true_self_grads)
    print("Linear self: Close: {}, Equal: {}".format(ratio_close, ratio_equal))
    product_equals = product_equals * ratio_equal

    ratio_close = check_isclose(neigh_grads, true_neigh_grads)
    ratio_equal = check_equal(neigh_grads, true_neigh_grads)
    print("Linear neigh: Close: {}, Equal: {}".format(ratio_close, ratio_equal))
    product_equals = product_equals * ratio_equal

    ratio_close = check_isclose(self_weight_grads, true_self_weight_grads)
    ratio_equal = check_equal(self_weight_grads, true_self_weight_grads)
    print("Linear self weight: Close: {}, Equal: {}".format(ratio_close, ratio_equal))
    product_equals = product_equals * ratio_equal

    ratio_close = check_isclose(self_bias_grads, true_self_bias_grads)
    ratio_equal = check_equal(self_bias_grads, true_self_bias_grads)
    print("Linear self bias: Close: {}, Equal: {}".format(ratio_close, ratio_equal))
    product_equals = product_equals * ratio_equal

    ratio_close = check_isclose(neigh_weight_grads, true_neigh_weight_grads)
    ratio_equal = check_equal(neigh_weight_grads, true_neigh_weight_grads)
    print("Linear neigh weight: Close: {}, Equal: {}".format(ratio_close, ratio_equal))
    product_equals = product_equals * ratio_equal

    ratio_close = check_isclose(neigh_bias_grads, true_neigh_bias_grads)
    ratio_equal = check_equal(neigh_bias_grads, true_neigh_bias_grads)
    print("Linear neigh bias: Close: {}, Equal: {}".format(ratio_close, ratio_equal))
    product_equals = product_equals * ratio_equal

    path = test_dir_path + "/value.npy"
    value = 1 if product_equals == 1.0 else 0
    save_return_value(value, path)


if __name__ == "__main__":
    test_sage_linear()
