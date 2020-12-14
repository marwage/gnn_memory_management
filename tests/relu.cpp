// Copyright 2020 Marcel Wagenländer

#include "activation.hpp"
#include "chunking.hpp"
#include "helper.hpp"
#include "tensors.hpp"

#include "catch2/catch.hpp"
#include <string>

const std::string home = std::getenv("HOME");
const std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data";
const std::string flickr_dir_path = dir_path + "/flickr";
const std::string test_dir_path = dir_path + "/tests";

int test_layer(Layer *layer, std::string py_name);
int test_layer_chunked(LayerChunked *layer, std::string py_name, long chunk_size);

int test_relu_double(long chunk_size) {
    std::string path;
    CudaHelper cuda_helper;

    // inputs matrices
    path = flickr_dir_path + "/features.npy";
    Matrix<float> features = load_npy_matrix<float>(path);
    Matrix<float> incoming_gradients(features.num_rows_, features.num_columns_, true);
    incoming_gradients.set_random_values();
    path = test_dir_path + "/incoming_gradients.npy";
    save_npy_matrix(&incoming_gradients, path);
    long num_nodes = features.num_rows_;
    long num_features = features.num_columns_;
    long num_chunks = ceil((float) num_nodes / (float) chunk_size);
    std::vector<Matrix<float>> features_chunked(num_chunks);
    std::vector<Matrix<float>> incoming_gradients_chunked(num_chunks);
    chunk_up(&features, &features_chunked, chunk_size);
    chunk_up(&incoming_gradients, &incoming_gradients_chunked, chunk_size);

    // layer
    ReluChunked relu(&cuda_helper, chunk_size, features.num_rows_, features.num_columns_);

    // forward
    std::vector<Matrix<float>> *output = relu.forward_double(&features_chunked);

    Matrix<float> output_one(num_nodes, num_features, false);
    stitch(output, &output_one);
    path = test_dir_path + "/output.npy";
    save_npy_matrix(&output_one, path);

    // backward
    std::vector<Matrix<float>> *gradients = relu.backward(&incoming_gradients_chunked);

    Matrix<float> gradients_one(num_nodes, num_features, false);
    stitch(gradients, &gradients_one);
    path = test_dir_path + "/gradients.npy";
    save_npy_matrix(&gradients_one, path);

    // test against Pytorch
    std::string command = "/home/ubuntu/gpu_memory_reduction/pytorch-venv/bin/python3 /home/ubuntu/gpu_memory_reduction/alzheimer/tests/relu.py";
    system(command.c_str());

    // read test result
    path = test_dir_path + "/value.npy";
    return read_return_value(path);
}

int test_relu_prop(long chunk_size) {
    std::string path;
    CudaHelper cuda_helper;

    // inputs matrices
    path = flickr_dir_path + "/features.npy";
    Matrix<float> features = load_npy_matrix<float>(path);
    Matrix<float> incoming_gradients(features.num_rows_, features.num_columns_, true);
    incoming_gradients.set_random_values();
    path = test_dir_path + "/incoming_gradients.npy";
    save_npy_matrix(&incoming_gradients, path);
    long num_nodes = features.num_rows_;
    long num_features = features.num_columns_;
    long num_chunks = ceil((float) num_nodes / (float) chunk_size);
    std::vector<Matrix<float>> features_chunked(num_chunks);
    std::vector<Matrix<float>> incoming_gradients_chunked(num_chunks);
    chunk_up(&features, &features_chunked, chunk_size);
    chunk_up(&incoming_gradients, &incoming_gradients_chunked, chunk_size);

    // layer
    ReluChunked relu(&cuda_helper, chunk_size, features.num_rows_, features.num_columns_);

    // forward
    std::vector<Matrix<float>> *output = relu.forward_prop(&features_chunked);

    Matrix<float> output_one(num_nodes, num_features, false);
    stitch(output, &output_one);
    path = test_dir_path + "/output.npy";
    save_npy_matrix(&output_one, path);

    // backward
    std::vector<Matrix<float>> *gradients = relu.backward(&incoming_gradients_chunked);

    Matrix<float> gradients_one(num_nodes, num_features, false);
    stitch(gradients, &gradients_one);
    path = test_dir_path + "/gradients.npy";
    save_npy_matrix(&gradients_one, path);

    // test against Pytorch
    std::string command = "/home/ubuntu/gpu_memory_reduction/pytorch-venv/bin/python3 /home/ubuntu/gpu_memory_reduction/alzheimer/tests/relu.py";
    system(command.c_str());

    // read test result
    path = test_dir_path + "/value.npy";
    return read_return_value(path);
}


TEST_CASE("ReLU", "[relu]") {
    Relu relu;
    CHECK(test_layer(&relu, "relu"));
}

TEST_CASE("ReLU, chunked", "[relu][chunked]") {
    ReluChunked relu;
    CHECK(test_layer_chunked(&relu, "relu", 1 << 15));
    CHECK(test_layer_chunked(&relu, "relu", 1 << 12));
    CHECK(test_layer_chunked(&relu, "relu", 1 << 8));
}

TEST_CASE("ReLU, chunked, double buffer", "[relu][chunked][doublebuffer]") {
    CHECK(test_relu_double(1 << 12));
    CHECK(test_relu_double(1 << 8));
    CHECK(test_relu_double(1 << 6));
}

TEST_CASE("ReLU, chunked, proposed", "[relu][chunked][doublebuffer][proposed]") {
    CHECK(test_relu_prop(1 << 12));
    CHECK(test_relu_prop(1 << 8));
    CHECK(test_relu_prop(1 << 6));
}
