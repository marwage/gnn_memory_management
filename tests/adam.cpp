// Copyright 2020 Marcel Wagenländer

#include "adam.hpp"
#include "cuda_helper.hpp"
#include "helper.hpp"
#include "sage_linear.hpp"

#include "catch2/catch.hpp"
#include <iostream>


int test_adam() {
    std::string home = std::getenv("HOME");
    std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data";
    std::string flickr_dir_path = dir_path + "/flickr";
    std::string test_dir_path = dir_path + "/tests";
    std::string path;

    long num_nodes = 2048;
    long num_in_features = 1024;
    long num_out_features = 512;
    float learning_rate = 0.003;
    long num_params = 2;

//    matrix<float> weight = gen_rand_matrix(num_in_features, num_out_features);
//    path = test_dir_path + "/weight.npy";
//    save_npy_matrix(weight, path);
//
//    matrix<float> bias = gen_rand_matrix(num_out_features, 1);
//    path = test_dir_path + "/bias.npy";
//    save_npy_matrix(bias, path);

    matrix<float> weight_grads = gen_rand_matrix(num_in_features, num_out_features);
    path = test_dir_path + "/weight_grads.npy";
    save_npy_matrix(weight_grads, path);

    matrix<float> bias_grads = gen_rand_matrix(num_out_features, 1);
    path = test_dir_path + "/bias_grads.npy";
    save_npy_matrix(bias_grads, path);

//    int num_params = 2;
//    matrix<float> **params = new matrix<float>*[num_params];
//    params[0] = &weight;
//    params[1] = &bias;

    CudaHelper cuda_helper;
    Linear linear(&cuda_helper, num_in_features, num_out_features, num_nodes);
    matrix<float> **parameters = linear.get_parameters();
    Adam adam(&cuda_helper, learning_rate, parameters, num_params);
//    linear.set_parameters(params);

    path = test_dir_path + "/weight.npy";
    save_npy_matrix(parameters[0], path);
    path = test_dir_path + "/bias.npy";
    save_npy_matrix(parameters[1], path);

    matrix<float> **grads = new matrix<float>*[num_params];
    grads[0] = &weight_grads;
    grads[1] = &bias_grads;

    matrix<float> *adam_gradients = adam.step(grads);

    linear.update_weights(adam_gradients);

    matrix<float> **params_updated = linear.get_parameters();

    path = test_dir_path + "/weight_updated.npy";
    save_npy_matrix(params_updated[0], path);
    path = test_dir_path + "/bias_updated.npy";
    save_npy_matrix(params_updated[1], path);

    char command[] = "/home/ubuntu/gpu_memory_reduction/pytorch-venv/bin/python3 /home/ubuntu/gpu_memory_reduction/alzheimer/tests/adam.py";
    system(command);

    path = test_dir_path + "/value.npy";
    return read_return_value(path);
}

TEST_CASE("Adam", "[adam]") {
    CHECK(test_adam());
}
