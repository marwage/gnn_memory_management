// Copyright 2020 Marcel Wagenländer

#include "cuda_helper.hpp"
#include "sage_linear.hpp"
#include "adam.hpp"
#include "helper.hpp"

#include <iostream>
#include "catch2/catch.hpp"


int test_adam() {
    std::string home = std::getenv("HOME");
    std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data";
    std::string flickr_dir_path = dir_path + "/flickr";
    std::string test_dir_path = dir_path + "/tests";
    std::string path;

    int rows = 1024;
    int columns = 512;

    float learning_rate = 0.003;

    matrix<float> weight;
    weight.rows = rows;
    weight.columns = columns;
    weight.values = reinterpret_cast<float *>(
            malloc(weight.rows * weight.columns * sizeof(float)));
    for (int i = 0; i < weight.rows * weight.columns; ++i) {
        weight.values[i] = rand();
    }
    path = test_dir_path + "/weight.npy";
    save_npy_matrix(weight, path);

    matrix<float> bias;
    bias.rows = columns;
    bias.columns = 1;
    bias.values = reinterpret_cast<float *>(
            malloc(bias.rows * bias.columns * sizeof(float)));
    for (int i = 0; i < bias.rows * bias.columns; ++i) {
        bias.values[i] = rand();
    }
    path = test_dir_path + "/bias.npy";
    save_npy_matrix(bias, path);

    matrix<float> weight_grads;
    weight_grads.rows = weight.rows;
    weight_grads.columns = weight.columns;
    weight_grads.values = reinterpret_cast<float *>(
            malloc(weight_grads.rows * weight_grads.columns * sizeof(float)));
    for (int i = 0; i < weight_grads.rows * weight_grads.columns; ++i) {
        weight_grads.values[i] = rand();
    }
    path = test_dir_path + "/weight_grads.npy";
    save_npy_matrix(weight_grads, path);

    matrix<float> bias_grads;
    bias_grads.rows = bias.rows;
    bias_grads.columns = bias.columns;
    bias_grads.values = reinterpret_cast<float *>(
                        malloc(bias_grads.rows * bias_grads.columns * sizeof(float)));
    for (int i = 0; i < bias_grads.rows * bias_grads.columns; ++i) {
        bias_grads.values[i] = rand();
    }
    path = test_dir_path + "/bias_grads.npy";
    save_npy_matrix(bias_grads, path);


    CudaHelper cuda_helper;
    int num_params = 2;
    matrix<float> *params = (matrix<float> *) malloc(num_params * sizeof(matrix<float>));
    params[0] = weight;
    params[1] = bias;
    Linear linear(columns, columns, &cuda_helper);
    linear.set_parameters(params);
    Adam adam(&cuda_helper, learning_rate, params, num_params);

    matrix<float> *grads = (matrix<float> *) malloc(num_params * sizeof(matrix<float>));
    grads[0] = weight_grads;
    grads[1] = bias_grads;
    matrix<float> *adam_gradients = adam.step(grads);

    linear.update_weights(adam_gradients);

    matrix<float> *params_updated = linear.get_parameters();

    path = test_dir_path + "/weight_updated.npy";
    save_npy_matrix(params_updated[0], path);
    path = test_dir_path + "/bias_updated.npy";
    save_npy_matrix(params_updated[1], path);

    char command[] = "/home/ubuntu/gpu_memory_reduction/pytorch-venv/bin/python3 /home/ubuntu/gpu_memory_reduction/alzheimer/tests/adam.py";
    system(command);

    path = test_dir_path + "/value.npy";
    return read_return_value(path);
}

TEST_CASE( "Adam", "[adam]" ) {
    CHECK(test_adam());
}
