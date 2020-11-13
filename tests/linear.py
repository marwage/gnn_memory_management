import numpy as np
import os
import scipy.io
import scipy.sparse as sp
import torch
from helper import check_equal, check_isclose, load_col_major, save_return_value, to_torch


def test_linear():
    product_equals = 1.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    home = os.getenv("HOME")
    dir_path = home + "/gpu_memory_reduction/alzheimer/data"
    flickr_dir_path = dir_path + "/flickr"
    test_dir_path = dir_path + "/tests"

    path = test_dir_path + "/input.npy"
    input_ = load_col_major(path)
    path = test_dir_path + "/in_gradients.npy"
    in_gradients = load_col_major(path)

    input_torch = to_torch(input_)
    in_gradients_torch = torch.from_numpy(in_gradients)
    in_gradients_torch = in_gradients_torch.to(device)

    path = test_dir_path + "/weight.npy"
    weight = load_col_major(path)
    path = test_dir_path + "/bias.npy"
    bias = load_col_major(path)
    path = test_dir_path + "/activations.npy"
    activations = load_col_major(path)

    weight_torch = to_torch(weight)
    bias_torch = to_torch(bias)

    true_activations_torch = torch.matmul(input_torch, weight_torch) + bias_torch.T

    true_activations = true_activations_torch.detach().cpu().numpy()

    ratio_close = check_isclose(activations, true_activations)
    ratio_equal = check_equal(activations, true_activations)
    print("SageLinear: Close: {}, Equal: {}".format(ratio_close, ratio_equal))
    product_equals = product_equals * ratio_equal

    # BACKPROPAGATION
    true_activations_torch.backward(in_gradients_torch)

    path = test_dir_path + "/input_gradients.npy"
    input_gradients = load_col_major(path)

    path = test_dir_path + "/weight_gradients.npy"
    weight_gradients = load_col_major(path)
    path = test_dir_path + "/bias_gradients.npy"
    bias_gradients = load_col_major(path)

    true_input_gradients = input_torch.grad.cpu().numpy()
    true_weight_gradients = weight_torch.grad.cpu().numpy()
    true_bias_gradients = bias_torch.grad.cpu().numpy()

    ratio_close = check_isclose(input_gradients, true_input_gradients)
    ratio_equal = check_equal(input_gradients, true_input_gradients)
    print("Input gradients: Close: {}, Equal: {}".format(ratio_close, ratio_equal))
    product_equals = product_equals * ratio_equal

    ratio_close = check_isclose(weight_gradients, true_weight_gradients)
    ratio_equal = check_equal(weight_gradients, true_weight_gradients)
    print("Weight gradients: Close: {}, Equal: {}".format(ratio_close, ratio_equal))
    product_equals = product_equals * ratio_equal

    ratio_close = check_isclose(bias_gradients, true_bias_gradients)
    ratio_equal = check_equal(bias_gradients, true_bias_gradients)
    print("Bias gradients: Close: {}, Equal: {}".format(ratio_close, ratio_equal))
    product_equals = product_equals * ratio_equal

    path = test_dir_path + "/value.npy"
    value = 1 if product_equals == 1.0 else 0
    save_return_value(value, path)


if __name__ == "__main__":
    test_linear()

