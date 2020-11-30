// Copyright 2020 Marcel Wagenländer

#include "cuda_helper.hpp"
#include "helper.hpp"
#include "sage_linear.hpp"
#include "tensors.hpp"

#include "catch2/catch.hpp"
#include <string>


int test_linear() {
    std::string home = std::getenv("HOME");
    std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data";
    std::string flickr_dir_path = dir_path + "/flickr";
    std::string test_dir_path = dir_path + "/tests";
    std::string path;

    int rows = 1024;
    int num_in_features = 512;
    int num_out_features = 256;

    Matrix<float> input(rows, num_in_features, true);
    input.set_values(true);
    path = test_dir_path + "/input.npy";
    save_npy_matrix(&input, path);

    Matrix<float> in_gradients(rows, num_out_features, true);
    in_gradients.set_values(true);
    path = test_dir_path + "/in_gradients.npy";
    save_npy_matrix(&in_gradients, path);

    CudaHelper cuda_helper;
    Linear linear(&cuda_helper, num_in_features, num_out_features, rows);

    Matrix<float> *activations = linear.forward(&input);
    Matrix<float> *input_gradients = linear.backward(&in_gradients);

    path = test_dir_path + "/activations.npy";
    save_npy_matrix(activations, path);
    path = test_dir_path + "/input_gradients.npy";
    save_npy_matrix(input_gradients, path);

    Matrix<float> **params = linear.get_parameters();
    Matrix<float> **weight_gradients = linear.get_gradients();
    path = test_dir_path + "/weight.npy";
    save_npy_matrix(params[0], path);
    path = test_dir_path + "/bias.npy";
    save_npy_matrix(params[1], path);
    path = test_dir_path + "/weight_gradients.npy";
    save_npy_matrix(weight_gradients[0], path);
    path = test_dir_path + "/bias_gradients.npy";
    save_npy_matrix(weight_gradients[1], path);

    char command[] = "/home/ubuntu/gpu_memory_reduction/pytorch-venv/bin/python3 /home/ubuntu/gpu_memory_reduction/alzheimer/tests/linear.py";
    system(command);

    path = test_dir_path + "/value.npy";
    return read_return_value(path);
}

int test_linear_set_parameters() {
    std::string home = std::getenv("HOME");
    std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data";
    std::string flickr_dir_path = dir_path + "/flickr";
    std::string test_dir_path = dir_path + "/tests";
    std::string path;

    long num_nodes = 2048;
    long num_in_features = 1024;
    long num_out_features = 512;
    long num_params = 2;

    Matrix<float> weight(num_in_features, num_out_features, false);
    weight.set_values(true);
    Matrix<float> bias(num_out_features, 1, false);
    bias.set_values(true);

    CudaHelper cuda_helper;
    Linear linear(&cuda_helper, num_in_features, num_out_features, num_nodes);

    Matrix<float> **parameters = new Matrix<float> *[num_params];
    parameters[0] = &weight;
    parameters[1] = &bias;

    linear.set_parameters(parameters);
    Matrix<float> **get_parameters = linear.get_parameters();

    compare_mat(parameters[0], get_parameters[0], "weights");
    compare_mat(parameters[1], get_parameters[1], "bias");
}


TEST_CASE("Linear", "[linear]") {
    CHECK(test_linear());
}

TEST_CASE("Linear, set parameters", "[linear][setparameters]") {
    CHECK(test_linear_set_parameters());
}