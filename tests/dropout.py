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


def num_equal_rows(A, B):
    num_rows = 0
    for i in range(A.shape[0]):
        if np.isclose(A[i], B[i]).sum() == A[i].size:
            num_rows = num_rows + 1
    return num_rows


def main():
    home = os.getenv("HOME")
    dir_path = home + "/gpu_memory_reduction/alzheimer/data"
    flickr_dir_path = dir_path + "/flickr"
    test_dir_path = dir_path + "/tests"

    path = flickr_dir_path + "/features.npy"
    features = np.load(path)
    path = test_dir_path + "/dropout_result.npy"
    dropout_result = load_col_major(path)

    probability = 0.2
    dropout_layer = torch.nn.Dropout(p=probability)

    features_torch = torch.from_numpy(features)
    features_torch.requires_grad_()
    features_torch.retain_grad()
    dropout_result_torch = dropout_layer(features_torch)
    true_dropout_result = dropout_result_torch.detach().numpy()  # Not really true

    ratio = check_isclose(dropout_result, true_dropout_result)
    print("Dropout: Ratio equal: {}".format(ratio))

    path = test_dir_path + "/dropout_gradients.npy"
    dropout_grads = load_col_major(path)
    path = test_dir_path + "/in_gradients.npy"
    in_gradients = load_col_major(path)

    in_gradients_torch = torch.from_numpy(in_gradients)
    dropout_result_torch.backward(in_gradients_torch)
    true_dropout_grads = features_torch.grad.numpy()

    ratio = check_isclose(dropout_grads, true_dropout_grads)
    print("Dropout gradients: Ratio equal: {}".format(ratio))


if __name__ == "__main__":
    main()
