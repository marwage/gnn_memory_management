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
    int columns = 512;
    int num_out_features = 256;

    matrix<float> input = gen_rand_matrix(rows, columns);
    path = test_dir_path + "/input.npy";
    save_npy_matrix(input, path);

    matrix<float> in_gradients = gen_rand_matrix(rows, num_out_features);
    path = test_dir_path + "/in_gradients.npy";
    save_npy_matrix(in_gradients, path);

    CudaHelper cuda_helper;
    Linear linear(columns, num_out_features, &cuda_helper);

    matrix<float> activations = linear.forward(input);
    matrix<float> input_gradients = linear.backward(in_gradients);

    path = test_dir_path + "/activations.npy";
    save_npy_matrix(activations, path);
    path = test_dir_path + "/input_gradients.npy";
    save_npy_matrix(input_gradients, path);

    matrix<float> *params = linear.get_parameters();
    matrix<float> *weight_gradients = linear.get_gradients();
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


TEST_CASE("Linear", "[linear]") {
    CHECK(test_linear());
}
