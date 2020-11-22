// Copyright 2020 Marcel Wagenländer

#include "graph_convolution.hpp"
#include "cuda_helper.hpp"
#include "helper.hpp"
#include "tensors.hpp"

#include "catch2/catch.hpp"

const std::string home = std::getenv("HOME");
const std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data";
const std::string flickr_dir_path = dir_path + "/flickr";
const std::string test_dir_path = dir_path + "/tests";


int test_graph_conv(matrix<float> input, sparse_matrix<float> adj, matrix<float> in_gradients) {
    std::string path;
    CudaHelper cuda_helper;
    GraphConvolution graph_conv(&cuda_helper, &adj, "mean", input.columns);

    matrix<float> activations = graph_conv.forward(input);

    path = test_dir_path + "/activations.npy";
    save_npy_matrix(activations, path);

    matrix<float> gradients = graph_conv.backward(in_gradients);

    path = test_dir_path + "/gradients.npy";
    save_npy_matrix(gradients, path);

    char command[] = "/home/ubuntu/gpu_memory_reduction/pytorch-venv/bin/python3 /home/ubuntu/gpu_memory_reduction/alzheimer/tests/graph_convolution.py";
    system(command);

    path = test_dir_path + "/value.npy";
    return read_return_value(path);
}


TEST_CASE("Graph convolution", "[graphconv]") {
    std::string path;
    path = flickr_dir_path + "/features.npy";
    matrix<float> features = load_npy_matrix<float>(path);
    to_column_major_inplace(&features);

    path = flickr_dir_path + "/adjacency.mtx";
    sparse_matrix<float> adjacency = load_mtx_matrix<float>(path);

    matrix<float> in_gradients = gen_rand_matrix(features.rows, features.columns);
    path = test_dir_path + "/in_gradients.npy";
    save_npy_matrix(in_gradients, path);

    CHECK(test_graph_conv(features, adjacency, in_gradients));
}
