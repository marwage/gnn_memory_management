import numpy as np
import os
import scipy.io
import scipy.sparse as sp
import torch


def load_col_major(path):
    mat = np.load(path)
    n, m = mat.shape
    mat = mat.reshape((m, n))
    mat = mat.transpose()

    return mat

def check_isclose(A, B):
    if (A.shape == B.shape):
        is_close = np.isclose(A, B)
        ratio_equal = is_close.sum() / B.size
    else:
        print(A.shape)
        print(B.shape)
        return 0

    return ratio_equal


def print_nan_coor(A):
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if np.isnan(A[i, j]):
                print("NaN at ({}, {})".format(i, j))

def num_close_rows(A, B):
    is_close = np.isclose(A, B)
    is_close_sum = is_close.sum(axis=1)
    close_rows = is_close_sum == A.shape[1]
    
    return close_rows.sum()

def print_small(A):
    print(A[0:3, 0:3])


def test_sage_linear_adam():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    home = os.getenv("HOME")
    dir_path = home + "/gpu_memory_reduction/alzheimer/data"
    flickr_dir_path = dir_path + "/flickr"
    test_dir_path = dir_path + "/tests"

    # FORWARD
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

    path = test_dir_path + "/self_weight.npy"
    self_weight = load_col_major(path)
    path = test_dir_path + "/self_bias.npy"
    self_bias = load_col_major(path)
    path = test_dir_path + "/neigh_weight.npy"
    neigh_weight = load_col_major(path)
    path = test_dir_path + "/neigh_bias.npy"
    neigh_bias = load_col_major(path)
    path = test_dir_path + "/activations.npy"
    activations = load_col_major(path)

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
    percentage_equal = check_isclose(activations, true_sage_linear_result)
    print("SageLinear: Percentage of equal elements: {}".format(percentage_equal))

    # BACKPROPAGATION
    in_gradients_torch = torch.from_numpy(in_gradients)
    in_gradients_torch = in_gradients_torch.to(device)
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

    ratio_equal = check_isclose(self_grads, true_self_grads)
    print("Linear self: Ratio of equal elements {}".format(ratio_equal))
    ratio_equal = check_isclose(neigh_grads, true_neigh_grads)
    print("Linear neigh: Ratio of equal elements {}".format(ratio_equal))
    ratio_equal = check_isclose(self_weight_grads, true_self_weight_grads)
    print("Linear self weight: Ratio {}".format(ratio_equal))
    ratio_equal = check_isclose(self_bias_grads, true_self_bias_grads)
    print("Linear self bias: Ratio {}".format(ratio_equal))
    #  print_small(self_bias_grads)
    #  print_small(true_self_bias_grads)
    ratio_equal = check_isclose(neigh_weight_grads, true_neigh_weight_grads)
    print("Linear neigh weight: Ratio {}".format(ratio_equal))
    ratio_equal = check_isclose(neigh_bias_grads, true_neigh_bias_grads)
    print("Linear neigh bias: Ratio {}".format(ratio_equal))
    #  print_small(neigh_bias_grads)
    #  print_small(true_neigh_bias_grads)

    # ADAM
    optimiser.step()

    path = test_dir_path + "/self_weight_updated.npy";
    self_weight_updated = load_col_major(path)
    path = test_dir_path + "/self_bias_updated.npy";
    self_bias_updated = load_col_major(path)
    path = test_dir_path + "/neigh_weight_updated.npy";
    neigh_weight_updated = load_col_major(path)
    path = test_dir_path + "/neigh_bias_updated.npy";
    neigh_bias_updated = load_col_major(path)

    true_self_weight = self_weight_torch.detach().cpu().numpy()
    ratio_equal = check_isclose(self_weight_updated, true_self_weight)
    print("Adam self weight: Ratio of equal elements {}".format(ratio_equal))
    true_self_bias = self_bias_torch.detach().cpu().numpy()
    ratio_equal = check_isclose(self_bias_updated, true_self_bias)
    print("Adam self bias: Ratio of equal elements {}".format(ratio_equal))
    print_small(self_bias_updated)
    print_small(true_self_bias)
    ratio_equal = check_isclose(neigh_weight_updated, neigh_weight_torch.detach().cpu().numpy())
    print("Adam neigh weight: Ratio of equal elements {}".format(ratio_equal))
    true_neigh_bias = neigh_bias_torch.detach().cpu().numpy()
    ratio_equal = check_isclose(neigh_bias_updated, true_neigh_bias) 
    print("Adam neigh bias: Ratio of equal elements {}".format(ratio_equal))

def compare_adam():
    home = os.getenv("HOME")
    dir_path = home + "/gpu_memory_reduction/alzheimer/data"
    flickr_dir_path = dir_path + "/flickr"
    test_dir_path = dir_path + "/tests"

    for i in range(4):
        path = test_dir_path + "/gradient_" + str(i) + ".npy"
        gradient = load_col_major(path)

        path = test_dir_path + "/gradient_chunked_" + str(i) + ".npy"
        gradient_chunked = load_col_major(path)

        ratio = check_isclose(gradient, gradient_chunked)
        print("Gradient of layer {}: Ratio {}".format(i, ratio))

        path = test_dir_path + "/weight_" + str(i) + ".npy"
        weight = load_col_major(path)

        path = test_dir_path + "/weight_chunked_" + str(i) + ".npy"
        weight_chunked = load_col_major(path)

        ratio = check_isclose(weight, weight_chunked)
        print("Weight of layer {}: Ratio {}".format(i, ratio))


if __name__ == "__main__":
    test_sage_linear_adam()

