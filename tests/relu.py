import numpy as np
import torch
import os


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


def main():
    home = os.getenv("HOME")
    dir_path = home + "/gpu_memory_reduction/alzheimer/data"
    flickr_dir_path = dir_path + "/flickr"
    test_dir_path = dir_path + "/tests"

    path = flickr_dir_path + "/features.npy"
    features = np.load(path)
    path = test_dir_path + "/relu_result.npy"
    relu_result = load_col_major(path)

    relu_layer = torch.nn.ReLU()

    features_torch = torch.from_numpy(features)
    features_torch.requires_grad_()
    features_torch.retain_grad()
    relu_result_torch = relu_layer(features_torch)
    true_relu_result = relu_result_torch.detach().numpy()

    ratio = check_isclose(relu_result, true_relu_result)
    print("ReLU: Ratio equal: {}".format(ratio))

    path = test_dir_path + "/relu_gradients.npy"
    relu_grads = load_col_major(path)
    path = test_dir_path + "/in_gradients.npy"
    in_gradients = load_col_major(path)

    in_gradients_torch = torch.from_numpy(in_gradients)
    relu_result_torch.backward(in_gradients_torch)
    true_relu_grads = features_torch.grad.numpy()

    ratio = check_isclose(relu_grads, true_relu_grads)
    print("ReLU gradients: Ratio equal: {}".format(ratio))


if __name__ == "__main__":
    main()
