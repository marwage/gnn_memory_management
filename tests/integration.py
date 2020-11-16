import numpy as np
import os
import scipy.io
import scipy.sparse as sp
import torch
from helper import (load_col_major, print_close_equal, check_close_equal,
        to_torch, print_not_close, write_return, update_return)


def integration_test():
    all_close = True
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # FORWARD PASS
    # check graph convolution
    path = test_dir_path + "/graph_convolution_result.npy"
    graph_conv_result = load_col_major(path)

    features_torch = to_torch(features)

    assert (sp.isspmatrix_coo(adj))
    values = adj.data
    indices = np.vstack((adj.row, adj.col))
    indices = torch.LongTensor(indices)
    values = torch.FloatTensor(values)
    shape = adj.shape
    adj_torch = torch.sparse.FloatTensor(indices, values, torch.Size(shape))
    adj_torch = adj_torch.to(device)

    graph_conv_result_torch = torch.sparse.mm(adj_torch, features_torch)

    # mean
    sum_torch = torch.sparse.sum(adj_torch, dim=-1)
    sum_torch = sum_torch.to_dense()
    for i in range(graph_conv_result_torch.shape[1]):
        graph_conv_result_torch[:, i] = graph_conv_result_torch[:, i] / sum_torch

    graph_conv_result_torch.requires_grad_()
    graph_conv_result_torch.retain_grad()

    true_graph_conv_result = graph_conv_result_torch.detach().cpu().numpy()

    ratio_close, ratio_equal = check_close_equal(graph_conv_result, true_graph_conv_result)
    all_close = update_return(ratio_close)
    print_close_equal("Graph convolution", ratio_close, ratio_equal)

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

    self_result_torch = torch.matmul(features_torch, self_weight_torch) + self_bias_torch.T
    neigh_result_torch = torch.matmul(graph_conv_result_torch, neigh_weight_torch) + neigh_bias_torch.T
    sage_linear_result_torch = self_result_torch + neigh_result_torch

    sage_linear_result_torch.requires_grad_()
    sage_linear_result_torch.retain_grad()

    true_sage_linear_result = sage_linear_result_torch.detach().cpu().numpy()

    ratio_close, ratio_equal = check_close_equal(sage_linear_result, true_sage_linear_result)
    all_close = update_return(ratio_close)
    print_close_equal("SageLinear", ratio_close, ratio_equal)

    # check relu
    path = test_dir_path + "/relu_result.npy"
    relu_result = load_col_major(path)

    relu_layer = torch.nn.ReLU()
    relu_result_torch = relu_layer(sage_linear_result_torch)

    relu_result_torch.requires_grad_()
    relu_result_torch.retain_grad()

    true_relu_result = relu_result_torch.detach().cpu().numpy()

    ratio_close, ratio_equal =check_close_equal(relu_result, true_relu_result)
    all_close = update_return(ratio_close)
    print_close_equal("ReLU", ratio_close, ratio_equal)

    # check log-softmax
    path = test_dir_path + "/log_softmax_result.npy"
    log_softmax_result = load_col_major(path)

    log_softmax_layer = torch.nn.LogSoftmax(dim=-1)
    log_softmax_result_torch = log_softmax_layer(relu_result_torch)

    log_softmax_result_torch.requires_grad_()
    log_softmax_result_torch.retain_grad()

    true_log_softmax_result = log_softmax_result_torch.detach().cpu().numpy()

    ratio_close, ratio_equal = check_close_equal(log_softmax_result, true_log_softmax_result)
    all_close = update_return(ratio_close)
    print_close_equal("Log-softmax", ratio_close, ratio_equal)

    # check loss
    path = test_dir_path + "/loss_result.npy"
    loss_result = np.load(path)
    loss_result = loss_result.squeeze()

    loss_layer = torch.nn.NLLLoss()
    classes_torch = torch.from_numpy(classes).long()
    classes_torch = classes_torch.to(device)
    loss_result_torch = loss_layer(log_softmax_result_torch, classes_torch)

    true_loss_result = loss_result_torch.detach().cpu().numpy()

    ratio_close, ratio_equal = check_close_equal(loss_result, true_loss_result)
    all_close = update_return(ratio_close)
    print_close_equal("Loss", ratio_close, ratio_equal)

    # BACKPROPAGATION
    # check loss
    path = test_dir_path + "/loss_grads.npy"
    loss_grads = load_col_major(path)

    loss_result_torch.backward()

    true_loss_grads = log_softmax_result_torch.grad.cpu().numpy()

    ratio_close, ratio_equal = check_close_equal(loss_grads, true_loss_grads)
    all_close = update_return(ratio_close)
    print_close_equal("Loss gradients", ratio_close, ratio_equal)

    # check log-softmax
    path = test_dir_path + "/log_softmax_grads.npy"
    log_softmax_grads = load_col_major(path)

    true_log_softmax_grads = relu_result_torch.grad.cpu().numpy()

    ratio_close, ratio_equal = check_close_equal(log_softmax_grads, true_log_softmax_grads)
    all_close = update_return(ratio_close)
    print_close_equal("Log-softmax gradients", ratio_close, ratio_equal)

    # check relu
    path = test_dir_path + "/relu_grads.npy"
    relu_grads = load_col_major(path)

    true_relu_grads = sage_linear_result_torch.grad.cpu().numpy()

    ratio_close, ratio_equal = check_close_equal(relu_grads, true_relu_grads)
    all_close = update_return(ratio_close)
    print_close_equal("ReLU gradients", ratio_close, ratio_equal)

    # check linear layer
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

    #  true_self_grads = features_torch.grad.cpu().numpy()
    true_neigh_grads = graph_conv_result_torch.grad.cpu().numpy()
    true_self_weight_grads = self_weight_torch.grad.cpu().numpy()
    true_self_bias_grads = self_bias_torch.grad.cpu().numpy()
    true_neigh_weight_grads = neigh_weight_torch.grad.cpu().numpy()
    true_neigh_bias_grads = neigh_bias_torch.grad.cpu().numpy()

    #  ratio_close, ratio_equal = check_close_equal(self_grads, true_self_grads)
    #  all_close = update_return(ratio_close)
    #  print_close_equal("SageLinear self gradients", ratio_close, ratio_equal)
    ratio_close, ratio_equal = check_close_equal(neigh_grads, true_neigh_grads)
    all_close = update_return(ratio_close)
    print_close_equal("SageLinear neigh gradients", ratio_close, ratio_equal)
    ratio_close, ratio_equal = check_close_equal(self_weight_grads, true_self_weight_grads)
    all_close = update_return(ratio_close)
    print_close_equal("SageLinear self weight", ratio_close , ratio_equal)
    ratio_close, ratio_equal = check_close_equal(self_bias_grads, true_self_bias_grads)
    all_close = update_return(ratio_close)
    print_close_equal("SageLinear self bias", ratio_close, ratio_equal)
    ratio_close, ratio_equal = check_close_equal(neigh_weight_grads, true_neigh_weight_grads)
    all_close = update_return(ratio_close)
    print_close_equal("SageLinear neigh weight", ratio_close, ratio_equal)
    ratio_close, ratio_equal = check_close_equal(neigh_bias_grads, true_neigh_bias_grads)
    all_close = update_return(ratio_close)
    print_close_equal("Linear neigh bias", ratio_close, ratio_equal)

    # check graph convolution
    #  path = test_dir_path + "/graph_convolution_grads.npy"
    #  graph_conv_grads = load_col_major(path)

    #  true_graph_conv_grads = features_torch.grad.cpu().numpy()

    #  ratio_close, ratio_equal = check_close_equal(graph_conv_grads, true_graph_conv_grads)
    #  if (ratio_close <= 0.98): all_close = False
    #  print_close_equal("Graph convolution gradients", ratio_close, ratio_equal)

    # check add
    path = test_dir_path + "/add_grads.npy"
    input_grads = load_col_major(path)
    true_input_grads = features_torch.grad.cpu().numpy()
    
    ratio_close, ratio_equal = check_close_equal(input_grads, true_input_grads)
    all_close = update_return(ratio_close)
    print_close_equal("Input gradients", ratio_close, ratio_equal)

    path = test_dir_path + "/value.npy"
    write_return(all_close, path)


if __name__ == "__main__":
    integration_test()

