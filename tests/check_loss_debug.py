import numpy as np
import os
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


home = os.getenv("HOME")
dir_path = home + "/gpu_memory_reduction/alzheimer/data"
flickr_dir_path = dir_path + "/flickr"
test_dir_path = dir_path + "/tests"

path = flickr_dir_path + "/classes.npy"
classes = np.load(path)

path = test_dir_path + "/log_softmax_result.npy"
log_softmax_result = load_col_major(path)

path = test_dir_path + "/loss_result.npy"
loss_result = np.load(path)
loss_result = loss_result.squeeze()

loss_layer = torch.nn.NLLLoss()
log_softmax_result_torch = torch.from_numpy(log_softmax_result)
log_softmax_result_torch.requires_grad_()
classes_torch = torch.from_numpy(classes).long()
true_loss_result_torch = loss_layer(log_softmax_result_torch, classes_torch)
true_loss_result = true_loss_result_torch.detach().numpy()

loss_diff = true_loss_result - loss_result
print("Loss: Difference between loss and true loss: {}".format(loss_diff))
percentage_off = loss_diff / true_loss_result
print("Loss: Percentage off: {}".format(percentage_off))

path = test_dir_path + "/loss_grads.npy"
loss_grads = load_col_major(path)
    
true_loss_result_torch.backward()

assert(log_softmax_result_torch.grad != None)
true_loss_grads = log_softmax_result_torch.grad.numpy()

ratio_equal = check_isclose(loss_grads, true_loss_grads)
print("Loss backward: Ratio of equal elements: {}".format(ratio_equal))

