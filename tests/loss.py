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
torch_log_softmax_out = torch.from_numpy(log_softmax_out)
torch_log_softmax_out.requires_grad_()
torch_classes = torch.from_numpy(classes).long()
true_loss = loss_layer(torch_log_softmax_out, torch_classes)
true_loss.backward()
true_loss = true_loss.detach().numpy()

print("Loss: {}".format(true_loss))

path = dir_path + "/loss_grads.npy"
loss_grads = np.load(path)

# to row-major
n, m = loss_grads.shape
loss_grads = loss_grads.reshape((m, n))
loss_grads = loss_grads.transpose()

torch_loss_grads = torch_log_softmax_out.grad.numpy()

assert (loss_grads.shape == torch_log_softmax_out.grad.shape)
is_close = np.isclose(loss_grads, torch_loss_grads)
percentage_equal = is_close.sum() / loss_grads.size
print("Loss backward: Percentage of equal elements: {}".format(percentage_equal))
