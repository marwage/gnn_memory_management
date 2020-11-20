// Copyright 2020 Marcel Wagenländer

#include "activation.hpp"
#include "helper.hpp"
#include "tensors.hpp"

#include "catch2/catch.hpp"
#include <string>


int test_relu(int chunk_size) {
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
    ReluParent *relu_layer;
    if (chunk_size == 0) {
        relu_layer = new Relu(&cuda_helper, features.rows, features.columns);
    } else {
        relu_layer = new ReluChunked(&cuda_helper, chunk_size, features.rows, features.columns);
    }

    matrix<float> activations = relu_layer->forward(features);
    path = test_dir_path + "/activations.npy";
    save_npy_matrix(activations, path);

    matrix<float> gradients = relu_layer->backward(in_gradients);
    path = test_dir_path + "/gradients.npy";
    save_npy_matrix(gradients, path);

    char command[] = "/home/ubuntu/gpu_memory_reduction/pytorch-venv/bin/python3 /home/ubuntu/gpu_memory_reduction/alzheimer/tests/relu.py";
    system(command);

    path = test_dir_path + "/value.npy";
    return read_return_value(path);
}


TEST_CASE("ReLU", "[relu]") {
    CHECK(test_relu(0));
}

TEST_CASE("ReLU, chunked", "[relu][chunked]") {
    CHECK(test_relu(1 << 15));
    CHECK(test_relu(1 << 12));
    CHECK(test_relu(1 << 8));
}
