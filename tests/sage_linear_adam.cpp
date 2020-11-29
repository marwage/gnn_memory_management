// Copyright 2020 Marcel Wagenländer

#include "adam.hpp"
#include "cuda_helper.hpp"
#include "helper.hpp"
#include "sage_linear.hpp"
#include "tensors.hpp"

#include "catch2/catch.hpp"
#include <string>

#include <iostream> // DEBUGGING


const std::string home = std::getenv("HOME");
const std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data";
const std::string flickr_dir_path = dir_path + "/flickr";
const std::string test_dir_path = dir_path + "/tests";


int test_sage_linear_adam(Matrix<float> *input_self, Matrix<float> *input_neigh, Matrix<float> *in_gradients, int chunk_size) {
    std::string path;
    float learning_rate = 0.003;

    CudaHelper cuda_helper;
    SageLinearParent *sage_linear_layer;
    if (chunk_size == 0) {// no chunking
        sage_linear_layer = new SageLinear(&cuda_helper, input_self->columns, in_gradients->columns, input_self->rows);
    } else {
        sage_linear_layer = new SageLinearChunked(&cuda_helper, input_self->columns, in_gradients->columns, chunk_size, input_self->rows);
    }
    Matrix<float> **params = sage_linear_layer->get_parameters();
    Adam adam(&cuda_helper, learning_rate, params, 4);

    save_params(params);

    Matrix<float> *activations = sage_linear_layer->forward(input_self, input_neigh);

    save_npy_matrix(activations, test_dir_path + "/activations.npy");

    SageLinearGradients *gradients = sage_linear_layer->backward(in_gradients);

    save_grads(gradients, sage_linear_layer->get_gradients());

    Matrix<float> *adam_gradients = adam.step(sage_linear_layer->get_gradients());
    sage_linear_layer->update_weights(adam_gradients);

    Matrix<float> **weights_updated = sage_linear_layer->get_parameters();
    save_npy_matrix(weights_updated[0], test_dir_path + "/self_weight_updated.npy");
    save_npy_matrix(weights_updated[1], test_dir_path + "/self_bias_updated.npy");
    save_npy_matrix(weights_updated[2], test_dir_path + "/neigh_weight_updated.npy");
    save_npy_matrix(weights_updated[3], test_dir_path + "/neigh_bias_updated.npy");

    char command[] = "/home/ubuntu/gpu_memory_reduction/pytorch-venv/bin/python3 /home/ubuntu/gpu_memory_reduction/alzheimer/tests/sage_linear_adam.py";
    system(command);

    path = test_dir_path + "/value.npy";
    return read_return_value(path);
}


TEST_CASE("SageLinear and Adam", "[sagelinear][adam]") {
    int rows = 1 << 15;
    int columns = 1 << 9;
    int num_out_features = 1 << 8;

    Matrix<float> input_self = gen_rand_matrix(rows, columns);
    save_npy_matrix(input_self, test_dir_path + "/input_self.npy");

    Matrix<float> input_neigh = gen_rand_matrix(rows, columns);
    save_npy_matrix(input_neigh, test_dir_path + "/input_neigh.npy");

    Matrix<float> in_gradients = gen_rand_matrix(rows, num_out_features);
    save_npy_matrix(in_gradients, test_dir_path + "/in_gradients.npy");

    CHECK(test_sage_linear_adam(&input_self, &input_neigh, &in_gradients, 0));
}

TEST_CASE("SageLinear and Adam chunked", "[sagelinear][adam][chunked]") {
    int rows = 1 << 15;
    int columns = 1 << 9;
    int num_out_features = 1 << 8;

    Matrix<float> input_self = gen_rand_matrix(rows, columns);
    save_npy_matrix(input_self, test_dir_path + "/input_self.npy");

    Matrix<float> input_neigh = gen_rand_matrix(rows, columns);
    save_npy_matrix(input_neigh, test_dir_path + "/input_neigh.npy");

    Matrix<float> in_gradients = gen_rand_matrix(rows, num_out_features);
    save_npy_matrix(in_gradients, test_dir_path + "/in_gradients.npy");

    CHECK(test_sage_linear_adam(&input_self, &input_neigh, &in_gradients, 1 << 15));
    CHECK(test_sage_linear_adam(&input_self, &input_neigh, &in_gradients, 1 << 12));
    CHECK(test_sage_linear_adam(&input_self, &input_neigh, &in_gradients, 1 << 8));
}
