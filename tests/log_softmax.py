import numpy as np
import os
import torch


home = os.getenv("HOME")
dir_path = home + "/gpu_memory_reduction/alzheimer/data/flickr"

path = dir_path + "/features.npy"
features = np.load(path)

path = dir_path + "/log_softmax_out.npy"
log_softmax_out = np.load(path)

# to row-major
n, m = log_softmax_out.shape
log_softmax_out = log_softmax_out.reshape((m, n))
log_softmax_out = log_softmax_out.transpose()

log_softmax_layer = torch.nn.LogSoftmax(dim=-1)
true_log_softmax_out = log_softmax_layer(torch.from_numpy(features))
true_log_softmax_out = true_log_softmax_out.numpy()

is_close = np.isclose(log_softmax_out, true_log_softmax_out)
percentage_equal = is_close.sum() / true_log_softmax_out.size
print("Log-softmax: Percentage of equal elements: {}".format(percentage_equal))

