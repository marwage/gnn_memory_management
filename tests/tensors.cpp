// Copyright 2020 Marcel Wagenländer

#include "tensors.hpp"

#include "catch2/catch.hpp"
#include <iostream>


int test_get_rows() {
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

    sparse_matrix<float> adj_rows = get_rows(adjacency, 5, 10);

    print_sparse_matrix(adj_rows);

    return 1;// TODO
}


TEST_CASE("Sparse matrix get rows", "[sparsematrix][getrows]") {
    CHECK(test_get_rows());
}
