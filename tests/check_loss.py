import numpy as np
import os
import torch


home = os.getenv("HOME")
dir_path = home + "/gpu_memory_reduction/alzheimer/data/flickr"

path = dir_path + "/log_softmax_out.npy"
log_softmax_out = np.load(path)

path = dir_path + "/classes.npy"
classes = np.load(path)

# to row-major
n, m = log_softmax_out.shape
log_softmax_out = log_softmax_out.reshape((m, n))
log_softmax_out = log_softmax_out.transpose()

loss_layer = torch.nn.NLLLoss()
true_loss = loss_layer(torch.from_numpy(log_softmax_out), torch.from_numpy(classes))
true_loss = true_loss.numpy()

print("Loss: {}".format(true_loss))
