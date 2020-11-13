// Copyright 2020 Marcel Wagenländer

#include "sage_linear.hpp"
#include "cuda_helper.hpp"
#include "tensors.hpp"
#include "helper.hpp"

#include <iostream>
#include <Python.h>

const std::string home = std::getenv("HOME");
const std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data";
const std::string flickr_dir_path = dir_path + "/flickr";
const std::string test_dir_path = dir_path + "/tests";


int test_sage_linear_forward() {
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

    path = test_dir_path + "/result.npy";
    save_npy_matrix(result, path);

    save_params(sage_linear.get_parameters());

    char command[] = "/home/ubuntu/gpu_memory_reduction/pytorch-venv/bin/python3 /home/ubuntu/gpu_memory_reduction/alzheimer/tests/sage_linear_forward.py";
    system(command);

    path = test_dir_path + "/value.npy";
    matrix<int> value = load_npy_matrix<int>(path);

    return value.values[0];
}

int test_sage_linear_forward_chunked(matrix<float> input_self, matrix<float> input_neigh, int chunk_size) {
    std::string path;

    CudaHelper cuda_helper;
    int num_nodes = input_self.rows;
    int num_out_features = 101;
    SageLinearChunked sage_linear_chunked(&cuda_helper, input_self.columns, num_out_features, chunk_size, num_nodes);

    matrix<float> result = sage_linear_chunked.forward(input_self, input_neigh);

    path = test_dir_path + "/result.npy";
    save_npy_matrix(result, path);

    save_params(sage_linear_chunked.get_parameters());

    char command[] = "/home/ubuntu/gpu_memory_reduction/pytorch-venv/bin/python3 /home/ubuntu/gpu_memory_reduction/alzheimer/tests/sage_linear_forward.py";
    system(command);

    path = test_dir_path + "/value.npy";
    matrix<int> value = load_npy_matrix<int>(path);

    return value.values[0];
}

int main() {
    std::string path;

//    int rows = 1009;
//    int columns = 211;
    int rows = 1024;
    int columns = 211;

    matrix<float> input_self = gen_rand_matrix(rows, columns);
    path = test_dir_path + "/input_self.npy";
    save_npy_matrix(input_self, path);

    matrix<float> input_neigh = gen_rand_matrix(rows, columns);
    path = test_dir_path + "/input_neigh.npy";
    save_npy_matrix(input_neigh, path);

    test_sage_linear_forward_chunked(input_self, input_neigh, 1024);
    test_sage_linear_forward_chunked(input_self, input_neigh, 512);
    test_sage_linear_forward_chunked(input_self, input_neigh, 256);
    test_sage_linear_forward_chunked(input_self, input_neigh, 128);
    test_sage_linear_forward_chunked(input_self, input_neigh, 64);
}
