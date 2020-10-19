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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.cuda.device(device):
        # FORWARD PASS
        # check dropout
        # to degree it is possible
        path = test_dir_path + "/dropout_result.npy"
        dropout_result = load_col_major(path)

        #  probability = 0.2
        #  dropout_layer = torch.nn.Dropout(p=probability)
        #  features_torch = torch.from_numpy(features)
        #  features_torch.requires_grad_()
        #  features_torch.retain_grad()
        #  dropout_result_torch = dropout_layer(features_torch)
        #  dropout_result_torch.requires_grad_()
        #  dropout_result_torch.retain_grad()
        #  true_dropout_result = dropout_result_torch.detach().numpy()  # Not really true

        #  percentage_equal = check_isclose(dropout_result, true_dropout_result)
        #  print("Dropout: Percentage of equal elements: {}".format(percentage_equal))

        dropout_result_torch = torch.from_numpy(dropout_result)
        dropout_result_torch.requires_grad_()
        dropout_result_torch.retain_grad()

        # check graph convolution
        path = test_dir_path + "/graph_convolution_result.npy"
        graph_conv_result = load_col_major(path)

        assert (sp.isspmatrix_coo(adj))
        values = adj.data
        indices = np.vstack((adj.row, adj.col))
        indices = torch.LongTensor(indices)
        values = torch.FloatTensor(values)
        shape = adj.shape
        adj_torch = torch.sparse.FloatTensor(indices, values, torch.Size(shape))

        graph_conv_result_torch = torch.sparse.mm(adj_torch, dropout_result_torch)
        graph_conv_result_torch.requires_grad_()
        graph_conv_result_torch.retain_grad()

        true_graph_conv_result = torch.sparse.mm(adj_torch,
                                                 torch.from_numpy(dropout_result)).numpy()

        # mean
        sum_torch = torch.sparse.sum(adj_torch, dim=-1)
        sum_torch = sum_torch.to_dense()
        sum_np = sum_torch.numpy()
        for i in range(graph_conv_result_torch.shape[1]):
            graph_conv_result_torch[:, i] = graph_conv_result_torch[:, i] / sum_torch
            true_graph_conv_result[:, i] = true_graph_conv_result[:, i] / sum_np

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

        self_weight_torch = torch.from_numpy(self_weight)
        self_weight_torch.requires_grad_()
        self_weight_torch.retain_grad()
        self_bias_torch = torch.from_numpy(self_bias)
        self_bias_torch.requires_grad_()
        self_bias_torch.retain_grad()

        neigh_weight_torch = torch.from_numpy(neigh_weight)
        neigh_weight_torch.requires_grad_()
        neigh_weight_torch.retain_grad()
        neigh_bias_torch = torch.from_numpy(neigh_bias)
        neigh_bias_torch.requires_grad_()
        neigh_bias_torch.retain_grad()

        learning_rate = 0.003
        params = [self_weight_torch,
            self_bias_torch,
            neigh_weight_torch,
            neigh_bias_torch]
        optimiser = torch.optim.Adam(params, lr=learning_rate)
        optimiser.zero_grad()

        self_result_torch = torch.matmul(dropout_result_torch, self_weight_torch) + self_bias_torch.T
        neigh_result_torch = torch.matmul(graph_conv_result_torch, neigh_weight_torch) + neigh_bias_torch.T
        true_sage_linear_result_torch = self_result_torch + neigh_result_torch

        true_sage_linear_result = true_sage_linear_result_torch.detach().numpy()

        percentage_equal = check_isclose(sage_linear_result, true_sage_linear_result)
        print("SageLinear: Percentage of equal elements: {}".format(percentage_equal))

        # check relu
        #  path = test_dir_path + "/relu_result.npy"
        #  relu_result = load_col_major(path)

        #  relu_layer = torch.nn.ReLU()
        #  sage_linear_result_torch = torch.from_numpy(sage_linear_result)
        #  sage_linear_result_torch.requires_grad_()
        #  sage_linear_result_torch = sage_linear_result_torch.retain_grad()
        #  true_relu_result = relu_layer(sage_linear_result_torch)
        #  true_relu_result = true_relu_result.detach().numpy()

        #  percentage_equal = check_isclose(relu_result, true_relu_result)
        #  print("ReLU: Percentage of equal elements: {}".format(percentage_equal))

        # check log-softmax
        path = test_dir_path + "/log_softmax_result.npy"
        log_softmax_result = load_col_major(path)

        log_softmax_layer = torch.nn.LogSoftmax(dim=-1)
        true_sage_linear_result_torch.requires_grad_()
        true_sage_linear_result_torch.retain_grad()
        true_log_softmax_result_torch = log_softmax_layer(true_sage_linear_result_torch)
        true_log_softmax_result_torch.requires_grad_()
        true_log_softmax_result_torch.retain_grad()
        true_log_softmax_result = true_log_softmax_result_torch.detach().numpy()

        percentage_equal = check_isclose(log_softmax_result, true_log_softmax_result)
        print("Log-softmax: Percentage of equal elements: {}".format(percentage_equal))

        # check loss
        path = test_dir_path + "/loss_result.npy"
        loss_result = np.load(path)
        loss_result = loss_result.squeeze()

        loss_layer = torch.nn.NLLLoss()
        classes_torch = torch.from_numpy(classes).long()
        true_loss_result_torch = loss_layer(true_log_softmax_result_torch, classes_torch)
        true_loss_result = true_loss_result_torch.detach().numpy()

        loss_diff = true_loss_result - loss_result
        print("Loss: Difference between loss and true loss: {}".format(loss_diff))
        ratio_off = loss_diff / true_loss_result
        print("Loss: Ratio off: {}".format(ratio_off))

        # BACKPROPAGATION
        # check loss
        path = test_dir_path + "/loss_grads.npy"
        loss_grads = load_col_major(path)

        true_loss_result_torch.backward()

        true_loss_grads = true_log_softmax_result_torch.grad.numpy()

        ratio_equal = check_isclose(loss_grads, true_loss_grads)
        print("Loss backward: Ratio of equal elements: {}".format(ratio_equal))

        # check log-softmax
        path = test_dir_path + "/log_softmax_grads.npy"
        log_softmax_grads = load_col_major(path)

        true_log_softmax_grads = true_sage_linear_result_torch.grad.numpy()

        ratio_equal = check_isclose(log_softmax_grads, true_log_softmax_grads)
        print("Log-softmax backward: Ratio of equal elements: {}".format(ratio_equal))

        # check relu
        path = test_dir_path + "/relu_grads.npy"
        relu_grads = load_col_major(path)
        # TODO last

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

        # true_self_grads NOT POSSIBLE
        true_neigh_grads = graph_conv_result_torch.grad.numpy()
        true_self_weight_grads = self_weight_torch.grad.numpy()
        true_self_bias_grads = self_bias_torch.grad.numpy()
        true_neigh_weight_grads = neigh_weight_torch.grad.numpy()
        true_neigh_bias_grads = neigh_bias_torch.grad.numpy()

        #  ratio_equal = check_isclose(self_grads, true_self_grads)
        #  print("Linear self: Ratio of equal elements {}".format(ratio_equal))
        ratio_equal = check_isclose(neigh_grads, true_neigh_grads)
        print("Linear neigh: Ratio of equal elements {}".format(ratio_equal))
        ratio_equal = check_isclose(self_weight_grads, true_self_weight_grads)
        print("Linear self weight: Ratio {}".format(ratio_equal))
        ratio_equal = check_isclose(self_bias_grads, true_self_bias_grads)
        print("Linear self bias: Ratio {}".format(ratio_equal))
        print("Linear self bias: !!! Currently, it looks good by the eye !!!")
        ratio_equal = check_isclose(neigh_weight_grads, true_neigh_weight_grads)
        print("Linear neigh weight: Ratio {}".format(ratio_equal))
        ratio_equal = check_isclose(neigh_bias_grads, true_neigh_bias_grads)
        print("Linear neigh bias: Ratio {}".format(ratio_equal))
        print("Linear neigh bias: !!! Currently, it looks good by the eye !!!")

        # check graph convolution
        path = test_dir_path + "/graph_convolution_grads.npy"
        graph_conv_grads = load_col_major(path)

        # NOT possible

        #  ratio_equal = check_isclose(graph_conv_grads, true_graph_conv_grads)
        #  print("Graph convolution: Ratio {}".format(ratio_equal))

        # check add
        path = test_dir_path + "/add_grads.npy"
        add_grads = load_col_major(path)

        true_add_grads = dropout_result_torch.grad.numpy()

        ratio_equal = check_isclose(add_grads, true_add_grads)
        print("Add gradients: Ratio {}".format(ratio_equal))

        # check Adam
        optimiser.step()


if __name__ == "__main__":
    test_computations()
