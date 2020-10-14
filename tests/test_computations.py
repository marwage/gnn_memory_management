import numpy as np
import scipy.sparse as sp
import scipy.io
import os
import torch


def load_col_major(path):
    mat = np.load(path)
    n, m = mat.shape
    mat = mat.reshape((m, n))
    mat = mat.transpose()

    return mat

def check_isclose(A, B):
    assert(A.shape == B.shape)
    is_close = np.isclose(A, B)
    percentage_equal = is_close.sum() / B.size

    return percentage_equal


def test_computations():
    home = os.getenv("HOME")
    dir_path = home + "/gpu_memory_reduction/alzheimer/data"
    flickr_dir_path = dir_path + "/flickr"
    test_dir_path = dir_path + "/tests"

    # read matrices
    path = flickr_dir_path + "/adjacency.mtx"
    adj = scipy.io.mmread(path)

    path = flickr_dir_path + "/features.npy"
    features = np.load(path)

    path = flickr_dir_path + "/classes.npy"
    classes = np.load(path)

    # FORWARD PASS
    # check dropout
    # to degree it is possible
    path = test_dir_path + "/dropout_result.npy"
    dropout_result = load_col_major(path)

    probability = 0.2
    dropout_layer = torch.nn.Dropout(p=probability)
    features_torch = torch.from_numpy(features)
    true_dropout_result_torch = dropout_layer(features_torch)
    true_dropout_result = true_dropout_result_torch.numpy()

    percentage_equal = check_isclose(dropout_result, true_dropout_result)
    print("Dropout: Percentage of equal elements: {}".format(percentage_equal))

    #check graph convolution
    path = test_dir_path + "/graph_convolution_result.npy"
    graph_conv_result = load_col_major(path)

    true_graph_conv_result = adj.dot(dropout_result)
    # mean
    ones = np.ones(adj.shape[0])
    sum = adj.dot(ones)
    for i in range(true_graph_conv_result.shape[1]):
        true_graph_conv_result[:, i] = true_graph_conv_result[:, i] / sum

    percentage_equal = check_isclose(graph_conv_result, true_graph_conv_result)
    print("Graph convolution: Percentage of equal elements: {}".format(percentage_equal))

    # check sage linear
    path = test_dir_path + "/self_weight.npy"
    self_weight = load_col_major(path)
    path = test_dir_path + "/self_bias.npy"
    self_bias = load_col_major(path)
    path = test_dir_path + "/neigh_weight.npy"
    neigh_weight = load_col_major(path)
    path = test_dir_path + "/neigh_bias.npy"
    neigh_bias = load_col_major(path)
    path = test_dir_path + "/linear_result.npy"
    sage_linear_result = load_col_major(path)

    self_result = dropout_result.dot(self_weight) + self_bias.T
    neigh_result = graph_conv_result.dot(neigh_weight) + neigh_bias.T
    true_sage_linear_result = self_result + neigh_result

    percentage_equal = check_isclose(sage_linear_result, true_sage_linear_result)
    print("SageLinear: Percentage of equal elements: {}".format(percentage_equal))
    
    # check relu
    path = test_dir_path + "/relu_result.npy"
    relu_result = load_col_major(path)

    relu_layer = torch.nn.ReLU()
    sage_linear_result_torch = torch.from_numpy(sage_linear_result)
    true_relu_result = relu_layer(sage_linear_result_torch)
    true_relu_result = true_relu_result.numpy()

    percentage_equal = check_isclose(relu_result, true_relu_result)
    print("ReLU: Percentage of equal elements: {}".format(percentage_equal))

    # check log-softmax
    path = test_dir_path + "/log_softmax_result.npy"
    log_softmax_result = load_col_major(path)

    log_softmax_layer = torch.nn.LogSoftmax(dim=-1)
    true_log_softmax_result_torch = log_softmax_layer(sage_linear_result_torch)
    true_log_softmax_result = true_log_softmax_result_torch.numpy()

    percentage_equal = check_isclose(log_softmax_result, true_log_softmax_result)
    print("Log-softmax: Percentage of equal elements: {}".format(percentage_equal))

    # check loss
    path = test_dir_path + "/loss_result.npy"
    loss_result = np.load(path)
    loss_result = loss_result.squeeze()

    loss_layer = torch.nn.NLLLoss()
    log_softmax_result_torch = torch.from_numpy(log_softmax_result)
    classes_torch = torch.from_numpy(classes).long()
    true_loss_result_torch = loss_layer(log_softmax_result_torch, classes_torch)
    true_loss_result = true_loss_result_torch.numpy()

    loss_diff = true_loss_result - loss_result
    print("Loss: Difference between loss and true loss: {}".format(loss_diff))
    percentage_off = loss_diff / true_loss_result
    print("Loss: Percentage off: {}".format(percentage_off))


if __name__ == "__main__":
    test_computations()

