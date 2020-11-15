// Copyright 2020 Marcel Wagenländer

#include "graph_convolution.hpp"
#include "tensors.hpp"
#include "cuda_helper.hpp"

#include "catch2/catch.hpp"


int test_graph_conv_chunked(int chunk_size) {
    std::string home = std::getenv("HOME");
    std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data";
    std::string flickr_dir_path = dir_path + "/flickr";
    std::string test_dir_path = dir_path + "/tests";

    // read features
    std::string path = flickr_dir_path + "/features.npy";
    matrix<float> features = load_npy_matrix<float>(path);
    to_column_major<float>(&features);

    // read adjacency
    path = flickr_dir_path + "/adjacency.mtx";
    sparse_matrix<float> adjacency = load_mtx_matrix<float>(path);

    CudaHelper cuda_helper;
    GraphConvChunked graph_conv_chunked(&cuda_helper, "mean", chunk_size);

    graph_conv_chunked.forward(adjacency, features);

    return 1; // TODO
}


TEST_CASE("Graph convolution chunked", "[graphconv][chunked]") {
    CHECK(test_graph_conv_chunked(1 << 12));
    CHECK(test_graph_conv_chunked(1 << 10));
    CHECK(test_graph_conv_chunked(1 << 8));
    CHECK(test_graph_conv_chunked(1 << 4));
}
