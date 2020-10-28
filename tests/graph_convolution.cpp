// Copyright 2020 Marcel Wagenländer

#include "graph_convolution.hpp"
#include "tensors.hpp"
#include "cuda_helper.hpp"


void check_chunked_graph_conv(int chunk_size) {
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
}

int main() {
    int chunk_size = 1 << 14;
    for (int i = 1; i < 10; ++i) {
        std::cout << "Chunk size " << i * chunk_size << std::endl;
        check_chunked_graph_conv(i * chunk_size);
    }
}
