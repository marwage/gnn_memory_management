// Copyright 2020 Marcel Wagenländer

#include "activation.hpp"
#include "cuda_helper.hpp"
#include "helper.hpp"
#include "tensors.hpp"

#include "catch2/catch.hpp"
#include <stdlib.h>


int test_log_softmax(int chunk_size) {
    std::string home = std::getenv("HOME");
    std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data";
    std::string flickr_dir_path = dir_path + "/flickr";
    std::string test_dir_path = dir_path + "/tests";

    std::string path = flickr_dir_path + "/features.npy";
    matrix<float> features = load_npy_matrix<float>(path);
    to_column_major_inplace<float>(&features);

    matrix<float> in_gradients = gen_rand_matrix(features.rows, features.columns);
    path = test_dir_path + "/in_gradients.npy";
    save_npy_matrix(in_gradients, path);

    CudaHelper cuda_helper;
    LogSoftmaxParent *log_softmax_layer;
    if (chunk_size == 0) {
        log_softmax_layer = new LogSoftmax(&cuda_helper, features.rows, features.columns);
    } else {
        log_softmax_layer = new LogSoftmaxChunked(&cuda_helper, chunk_size, features.rows, features.columns);
    }

    matrix<float> activations = log_softmax_layer->forward(features);
    path = test_dir_path + "/activations.npy";
    save_npy_matrix(activations, path);

    matrix<float> gradients = log_softmax_layer->backward(in_gradients);
    path = test_dir_path + "/gradients.npy";
    save_npy_matrix(gradients, path);

    char command[] = "/home/ubuntu/gpu_memory_reduction/pytorch-venv/bin/python3 /home/ubuntu/gpu_memory_reduction/alzheimer/tests/log_softmax.py";
    system(command);

    path = test_dir_path + "/value.npy";
    return read_return_value(path);
}


TEST_CASE("Log-softmax", "[logsoftmax]") {
    CHECK(test_log_softmax(0));
}

TEST_CASE("Log-softmax, chunked", "[logsoftmax][chunked]") {
    CHECK(test_log_softmax(1 << 15));
    CHECK(test_log_softmax(1 << 12));
    CHECK(test_log_softmax(1 << 8));
}
