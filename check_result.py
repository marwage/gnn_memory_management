import json
import numpy as np
import scipy.sparse as sp
import scipy.io
import os
import torch


home = os.getenv("HOME")
dir_path = home + "/gpu_memory_reduction/alzheimer/data/flickr"

# read matrices
path = dir_path + "/adjacency.mtx"
adj = scipy.io.mmread(path)

path = dir_path + "/features.npy"
features = np.load(path)

# check read/write
check_write = False
if check_write:
    path = dir_path + "/features_write.npy";
    features_write = np.load(path);

    assert(features_write.shape == features.shape)
    percentage_equal = (features_write == features).sum() / features.size

    print("Write")
    print("Percentage of equal elements: {}".format(percentage_equal))

# check dropout
# to degree it is possible
check_dropout = False
if check_dropout:
    path = dir_path + "/dropout_result.npy"
    dropout_result = np.load(path)

    probability = 0.2
    dropout_layer = torch.nn.Dropout(p=probability)
    features_torch = torch.from_numpy(features)
    torch_dropout_result = dropout_layer(features_torch)
    torch_dropout_result = torch_dropout_result.numpy()

    assert(dropout_result.shape == torch_dropout_result.shape)
    percentage_equal = (dropout_result == torch_dropout_result).sum() / dropout_result.size

    print("Dropout")
    print("Percentage of equal elements: {}".format(percentage_equal))

# check transposed features
check_transposed_features = False
if check_transposed_features:
    path = dir_path + "/features_T.npy"
    features_T = np.load(path)
    features_T_true = features.T

    print("features_T shape {}".format(features_T.shape))
    print("features_T_true shape {}".format(features_T_true.shape))

    assert(features_T.shape == features_T_true.shape)
    percentage_equal = (features_T == features_T_true).sum() / features_T_true.size

    print("Transposed features")
    print("Percentage of equal elements: {}".format(percentage_equal))

#check graph convolution
check_graph_conv = False
if check_graph_conv:
    path = dir_path + "/graph_conv_result.npy"
    graph_conv_result = np.load(path)

    graph_conv_true_result = adj.dot(features)

    assert(graph_conv_result.shape == graph_conv_true_result.shape)
    percentage_equal = (graph_conv_result == graph_conv_true_result).sum() / graph_conv_true_result.size
    print("Graph convolution")
    print("Percentage of equal elements: {}".format(percentage_equal))

# check graph convolution with mean
check_graph_conv_mean = False
if check_graph_conv_mean:
    path = dir_path + "/graph_conv_mean_result.npy"
    graph_conv_mean_result = np.load(path)
    path = dir_path + "/sum.npy"
    sum = np.load(path)
    sum = np.squeeze(sum)

    true_graph_conv_mean_result = adj.dot(features)
    true_sum = adj.dot(np.ones((adj.shape[0],)))
    for i in range(true_graph_conv_mean_result.shape[1]):
        true_graph_conv_mean_result[:, i] = true_graph_conv_mean_result[:, i] / true_sum

    assert(sum.shape == true_sum.shape)

    assert(graph_conv_mean_result.shape == true_graph_conv_mean_result.shape)
    is_close = np.isclose(graph_conv_mean_result, true_graph_conv_mean_result)
    percentage_equal = is_close.sum() / true_graph_conv_mean_result.size
    print("Graph convolution with mean")
    print("Percentage of equal elements: {}".format(percentage_equal))

# check relu
check_relu = True
if check_relu:
    path = dir_path + "/relu_result.npy"
    relu_result = np.load(path)
    
    relu_layer = torch.nn.functional.relu
    features_torch = torch.from_numpy(features)
    true_relu_result = relu_layer(features_torch)
    true_relu_result = true_relu_result.numpy()

    assert(relu_result.shape == true_relu_result.shape)
    is_close = np.isclose(relu_result, true_relu_result)
    percentage_equal = is_close.sum() / true_relu_result.size
    print("ReLU")
    print("Percentage of equal elements: {}".format(percentage_equal))

    

