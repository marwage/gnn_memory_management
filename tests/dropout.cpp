// Copyright 2020 Marcel Wagenländer

#include <iostream>

#include "dropout.hpp"
#include "tensors.hpp"


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
    Dropout dropout_layer(&cuda_helper);

    matrix<float> dropout_result = dropout_layer.forward(features);
    path = test_dir_path + "/dropout_result.npy";
    save_npy_matrix(dropout_result, path);

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

    matrix<float> gradients = dropout_layer.backward(in_gradients);
    path = test_dir_path + "/dropout_gradients.npy";
    save_npy_matrix(gradients, path);

    // compare with Pytorch, Numpy, SciPy
    char command[] = "/home/ubuntu/gpu_memory_reduction/pytorch-venv/bin/python3 /home/ubuntu/gpu_memory_reduction/alzheimer/tests/dropout.py";
    system(command);
}

int num_equal_rows(matrix<float> A, matrix<float> B) {
    int num_rows = 0;
    bool equal_row = true;

    for (int i = 0; i < A.rows; ++i) {
        equal_row = true;
        for (int j = 0; j < A.columns; ++j) {
            if (A.values[j * A.rows + i] != B.values[j * A.rows + i]) {
                equal_row = false;
            }
        }
        if (equal_row) {
            num_rows = num_rows + 1;
        }
    }

    return num_rows;
}

void check_dropout_chunked(int chunk_size) {
    std::string home = std::getenv("HOME");
    std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data";
    std::string flickr_dir_path = dir_path + "/flickr";
    std::string test_dir_path = dir_path + "/tests";

    // read features
    std::string path = flickr_dir_path + "/features.npy";
    matrix<float> features = load_npy_matrix<float>(path);
    to_column_major<float>(&features);

    CudaHelper cuda_helper;
    Dropout dropout_layer(&cuda_helper);

    matrix<float> dropout_result = dropout_layer.forward_chunked(features, chunk_size);
    path = test_dir_path + "/dropout_result_chunked.npy";
    save_npy_matrix(dropout_result, path);

    path = test_dir_path + "/in_gradients.npy";
    matrix<float> in_gradients = load_npy_matrix<float>(path);

    matrix<float> gradients = dropout_layer.backward_chunked(in_gradients, chunk_size);
    path = test_dir_path + "/dropout_gradients_chunked.npy";
    save_npy_matrix(gradients, path);

    char command[] = "/home/ubuntu/gpu_memory_reduction/pytorch-venv/bin/python3 /home/ubuntu/gpu_memory_reduction/alzheimer/tests/dropout_chunked.py";
    system(command);
}

int main() {
    check_dropout();
    int chunk_size = 1 << 14;
    int current_chunk_size;
    for (int i = 1; i < 10; ++i) {
        current_chunk_size = i * chunk_size;
        std::cout << "Chunk size " << current_chunk_size << std::endl;
        check_dropout_chunked(current_chunk_size);
    }
}

