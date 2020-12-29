// Copyright 2020 Marcel Wagenländer

#include "chunking.hpp"
#include "cuda_helper.hpp"
#include "feature_aggregation.hpp"
#include "helper.hpp"
#include "tensors.hpp"

#include "catch2/catch.hpp"


const std::string home = std::getenv("HOME");
const std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data";
const std::string test_dir_path = dir_path + "/tests";
const std::string flickr_dir_path = dir_path + "/flickr";
const std::string products_dir_path = dir_path + "/products";


int test_graph_conv(Matrix<float> *input, SparseMatrix<float> *adj, Matrix<float> *in_gradients) {
    std::string path;
    CudaHelper cuda_helper;
    FeatureAggregation graph_conv(&cuda_helper, adj, "mean", input->num_rows_, input->num_columns_);

    Matrix<float> *activations = graph_conv.forward(input);

    check_nans(activations, "activations");

    Matrix<float> *gradients = graph_conv.backward(in_gradients);

    check_nans(gradients, "Gradients");

    path = test_dir_path + "/activations.npy";
    save_npy_matrix(activations, path);

    path = test_dir_path + "/gradients.npy";
    save_npy_matrix(gradients, path);

    char command[] = "/home/ubuntu/gpu_memory_reduction/pytorch-venv/bin/python3 /home/ubuntu/gpu_memory_reduction/alzheimer/tests/graph_convolution.py";
    system(command);

    path = test_dir_path + "/value.npy";
    return read_return_value(path);
}

int test_graph_conv_chunked(FeatureAggregationChunked *graph_convolution, long chunk_size) {
    std::string path;
    path = flickr_dir_path + "/features.npy";
    Matrix<float> features = load_npy_matrix<float>(path);

    path = flickr_dir_path + "/adjacency.mtx";
    SparseMatrix<float> adjacency = load_mtx_matrix<float>(path);

    Matrix<float> incoming_gradients(features.num_rows_, features.num_columns_, false);
    incoming_gradients.set_random_values();
    path = test_dir_path + "/in_gradients.npy";
    save_npy_matrix(&incoming_gradients, path);

    CudaHelper cuda_helper;
    graph_convolution->set(&cuda_helper, &adjacency, "mean", features.num_columns_, chunk_size, features.num_rows_);

    long num_nodes = features.num_rows_;
    long num_features = features.num_columns_;
    long num_chunks = ceil((float) num_nodes / (float) chunk_size);
    std::vector<Matrix<float>> features_chunked(num_chunks);
    chunk_up(&features, &features_chunked, chunk_size);
    std::vector<Matrix<float>> incoming_gradients_chunked(num_chunks);
    chunk_up(&incoming_gradients, &incoming_gradients_chunked, chunk_size);

    std::vector<Matrix<float>> *activations = graph_convolution->forward(&features_chunked);

    check_nans(activations, "activations");

    std::vector<Matrix<float>> *gradients = graph_convolution->backward(&incoming_gradients_chunked);

    Matrix<float> activations_one(num_nodes, num_features, true);
    stitch(activations, &activations_one);
    path = test_dir_path + "/activations.npy";
    save_npy_matrix(&activations_one, path);

    Matrix<float> gradients_one(num_nodes, num_features, true);
    stitch(gradients, &gradients_one);
    path = test_dir_path + "/gradients.npy";
    save_npy_matrix(&gradients_one, path);

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
    FeatureAggregationChunked graph_convolution;
    CHECK(test_graph_conv_chunked(&graph_convolution, 1 << 15));
    CHECK(test_graph_conv_chunked(&graph_convolution, 1 << 14));
    CHECK(test_graph_conv_chunked(&graph_convolution, 1 << 13));
}

TEST_CASE("Graph convolution, pipelined", "[graphconv][pipelined]") {
    FeatureAggregationPipelined graph_convolution;
    CHECK(test_graph_conv_chunked(&graph_convolution, 1 << 15));
    CHECK(test_graph_conv_chunked(&graph_convolution, 1 << 14));
    CHECK(test_graph_conv_chunked(&graph_convolution, 1 << 13));
}
