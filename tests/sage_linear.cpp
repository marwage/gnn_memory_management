// Copyright 2020 Marcel Wagenländer

#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"

#include "sage_linear.hpp"
#include "cuda_helper.hpp"
#include "tensors.hpp"
#include "helper.hpp"

#include <iostream>
#include <Python.h>


int test_sage_linear() {
    std::string home = std::getenv("HOME");
    std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data";
    std::string flickr_dir_path = dir_path + "/flickr";
    std::string test_dir_path = dir_path + "/tests";
    std::string path;

    int rows = 1024;
    int columns = 512;
    int num_out_features = 256;

    matrix<float> input_self = gen_rand_matrix(rows, columns);
    path = test_dir_path + "/input_self.npy";
    save_npy_matrix(input_self, path);

    matrix<float> input_neigh = gen_rand_matrix(rows, columns);
    path = test_dir_path + "/input_neigh.npy";
    save_npy_matrix(input_neigh, path);

    matrix<float> in_gradients = gen_rand_matrix(rows, num_out_features);
    path = test_dir_path + "/in_gradients.npy";
    save_npy_matrix(in_gradients, path);

    CudaHelper cuda_helper;
    SageLinear sage_linear(columns, num_out_features, &cuda_helper);

    matrix<float> result = sage_linear.forward(input_self, input_neigh);
    SageLinear::SageLinearGradients gradients = sage_linear.backward(in_gradients);

    path = test_dir_path + "/result.npy";
    save_npy_matrix(result, path);

    save_params(sage_linear.get_parameters());

    save_grads(&gradients, sage_linear.get_gradients());

    char command[] = "/home/ubuntu/gpu_memory_reduction/pytorch-venv/bin/python3 /home/ubuntu/gpu_memory_reduction/alzheimer/tests/sage_linear.py";
    system(command);

    path = test_dir_path + "/value.npy";
    matrix<int> value = load_npy_matrix<int>(path);

    return value.values[0];
}

int test_sage_linear_chunked() {
    std::string home = std::getenv("HOME");
    std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data";
    std::string flickr_dir_path = dir_path + "/flickr";
    std::string test_dir_path = dir_path + "/tests";
    std::string path;

    int rows = 1024;
    int columns = 512;
    int num_out_features = 256;

    matrix<float> input_self = gen_rand_matrix(rows, columns);
    path = test_dir_path + "/input_self.npy";
    save_npy_matrix(input_self, path);

    matrix<float> input_neigh = gen_rand_matrix(rows, columns);
    path = test_dir_path + "/input_neigh.npy";
    save_npy_matrix(input_neigh, path);

    matrix<float> in_gradients = gen_rand_matrix(rows, num_out_features);
    path = test_dir_path + "/in_gradients.npy";
    save_npy_matrix(in_gradients, path);

    CudaHelper cuda_helper;
    int chunk_size = 128;
    int num_nodes = rows;
    SageLinearChunked sage_linear_chunked(&cuda_helper, columns, num_out_features, chunk_size, num_nodes);

    matrix<float> result = sage_linear_chunked.forward(input_self, input_neigh);
    SageLinear::SageLinearGradients gradients = sage_linear_chunked.backward(in_gradients);

    path = test_dir_path + "/result.npy";
    save_npy_matrix(result, path);

    save_params(sage_linear_chunked.get_parameters());

    save_grads(&gradients, sage_linear_chunked.get_gradients());

    char command[] = "/home/ubuntu/gpu_memory_reduction/pytorch-venv/bin/python3 /home/ubuntu/gpu_memory_reduction/alzheimer/tests/sage_linear.py";
    system(command);

    path = test_dir_path + "/value.npy";
    matrix<int> value = load_npy_matrix<int>(path);

    return value.values[0];
}


TEST_CASE( "SageLinear", "[sagelinear]" ) {
    CHECK( test_sage_linear() );
}

TEST_CASE( "SageLinearChunked", "[sagelinear][chunked]" ) {
    CHECK( test_sage_linear_chunked() );
}
