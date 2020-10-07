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

path = dir_path + "/graph_convolution_result.npy"
graph_conv_result = np.load(path)

# graph convolution with scipy
true_result = adj.dot(features)

assert(graph_conv_result.shape == true_result.shape)
percentage_equal = (graph_conv_result == true_result).sum() / graph_conv_result.size
print("Graph convolution")
print("Percentage of equal elements: {}".format(percentage_equal))

# read dropout result
path = dir_path + "/dropout_result.npy"
dropout_result = np.load(path)

# dropout with torch
probability = 0.2
dropout_layer = torch.nn.Dropout(p=probability)
features_torch = torch.from_numpy(features)
torch_dropout_result = dropout_layer(features_torch)

percentage_equal = (dropout_result == torch_dropout_result.numpy()).sum() / dropout_result.size
print("Dropout")
print("Percentage of equal elements: {}".format(percentage_equal))

