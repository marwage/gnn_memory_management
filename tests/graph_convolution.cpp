// Copyright 2020 Marcel Wagenländer

#include "graph_convolution.hpp"
#include "cuda_helper.hpp"
#include "helper.hpp"
#include "tensors.hpp"
#include "chunk.hpp"

#include "catch2/catch.hpp"


const std::string home = std::getenv("HOME");
const std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data";
const std::string test_dir_path = dir_path + "/tests";
const std::string flickr_dir_path = dir_path + "/flickr";
const std::string products_dir_path = dir_path + "/products";


int test_graph_conv(Matrix<float> *input, SparseMatrix<float> *adj, Matrix<float> *in_gradients) {
    std::string path;
    CudaHelper cuda_helper;
    GraphConvolution graph_conv(&cuda_helper, adj, "mean", input->num_rows_, input->num_columns_);

    Matrix<float> *activations = graph_conv.forward(input);

    check_nans(activations, "activations");

    Matrix<float> *gradients = graph_conv.backward(in_gradients);

    path = test_dir_path + "/activations.npy";
    save_npy_matrix(activations, path);

    path = test_dir_path + "/gradients.npy";
    save_npy_matrix(gradients, path);

    char command[] = "/home/ubuntu/gpu_memory_reduction/pytorch-venv/bin/python3 /home/ubuntu/gpu_memory_reduction/alzheimer/tests/graph_convolution.py";
    system(command);

    path = test_dir_path + "/value.npy";
    return read_return_value(path);
}

int test_graph_conv_chunked(Matrix<float> *input, SparseMatrix<float> *adj, Matrix<float> *in_gradients, long chunk_size) {
    std::string path;
    CudaHelper cuda_helper;
    GraphConvChunked graph_conv(&cuda_helper, adj, "mean", input->num_columns_, chunk_size, input->num_rows_);

    long num_chunks = ceil((float) input->num_rows_ / (float) chunk_size);
    std::vector<Matrix<float>> input_chunked(num_chunks);
    chunk_up(input, &input_chunked, chunk_size);

    Matrix<float> *activations = graph_conv.forward(&input_chunked);

    check_nans(activations, "activations");

    Matrix<float> *gradients = graph_conv.backward(in_gradients);

    path = test_dir_path + "/activations.npy";
    save_npy_matrix(activations, path);

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
    Matrix<float> features = load_npy_matrix<float>(path);
    to_column_major_inplace(&features);

    path = flickr_dir_path + "/adjacency.mtx";
    SparseMatrix<float> adjacency = load_mtx_matrix<float>(path);

    Matrix<float> in_gradients(features.num_rows_, features.num_columns_, false);
    in_gradients.set_random_values();
    path = test_dir_path + "/in_gradients.npy";
    save_npy_matrix(&in_gradients, path);

    CHECK(test_graph_conv(&features, &adjacency, &in_gradients));
}

TEST_CASE("Graph convolution, chunked", "[graphconv][chunked]") {
    std::string path;
    path = flickr_dir_path + "/features.npy";
    Matrix<float> features = load_npy_matrix<float>(path);

    path = flickr_dir_path + "/adjacency.mtx";
    SparseMatrix<float> adjacency = load_mtx_matrix<float>(path);

    Matrix<float> in_gradients(features.num_rows_, features.num_columns_, true);
    in_gradients.set_random_values();
    path = test_dir_path + "/in_gradients.npy";
    save_npy_matrix(&in_gradients, path);

    CHECK(test_graph_conv_chunked(&features, &adjacency, &in_gradients, 1 << 15));
    CHECK(test_graph_conv_chunked(&features, &adjacency, &in_gradients, 1 << 14));
    CHECK(test_graph_conv_chunked(&features, &adjacency, &in_gradients, 1 << 13));
}
