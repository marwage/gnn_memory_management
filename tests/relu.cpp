// Copyright 2020 Marcel Wagenländer

#include <iostream>

#include "tensors.hpp"
#include "activation.hpp"


void check_dropout() {
    std::string home = std::getenv("HOME");
    std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data";
    std::string flickr_dir_path = dir_path + "/flickr";
    std::string test_dir_path = dir_path + "/tests";

    // read features
    std::string path = flickr_dir_path + "/features.npy";
    matrix<float> features = load_npy_matrix<float>(path);
    to_column_major<float>(&features);

    CudaHelper cuda_helper;
    Relu relu_layer(&cuda_helper);

    matrix<float> relu_result = relu_layer.forward(features);
    path = test_dir_path + "/relu_result.npy";
    save_npy_matrix(relu_result, path);

    matrix<float> in_gradients;
    in_gradients.rows = features.rows;
    in_gradients.columns = features.columns;
    in_gradients.values = reinterpret_cast<float *>(
            malloc(in_gradients.rows * in_gradients.columns * sizeof(float)));
    for (int i = 0; i < in_gradients.rows * in_gradients.columns; ++i) {
        in_gradients.values[i] = rand();
    }
    path = test_dir_path + "/in_gradients.npy";
    save_npy_matrix(in_gradients, path);

    matrix<float> gradients = relu_layer.backward(in_gradients);
    path = test_dir_path + "/relu_gradients.npy";
    save_npy_matrix(gradients, path);

    // compare with Pytorch, Numpy, SciPy
    char command[] = "/home/ubuntu/gpu_memory_reduction/pytorch-venv/bin/python3 /home/ubuntu/gpu_memory_reduction/alzheimer/tests/relu.py";
    system(command);
}

int main() {
    check_dropout();
}