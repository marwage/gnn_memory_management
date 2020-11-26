// Copyright 2020 Marcel Wagenländer

#include "helper.hpp"
#include "cuda_helper.hpp"
#include "tensors.hpp"

#include "catch2/catch.hpp"
#include <string>


int test_sparse_transpose() {
    std::string home = std::getenv("HOME");
    std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data";
    std::string flickr_dir_path = dir_path + "/flickr";
    std::string test_dir_path = dir_path + "/tests";
    std::string path;

    CudaHelper cuda_helper;

    path = flickr_dir_path + "/adjacency.mtx";
    sparse_matrix<float> adjacency = load_mtx_matrix<float>(path);

    transpose_csr_matrix(&adjacency, &cuda_helper);

    return 1; // TODO
}

int test_get_rows() {
    std::string home = std::getenv("HOME");
    std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data";
    std::string flickr_dir_path = dir_path + "/flickr";
    std::string test_dir_path = dir_path + "/tests";
    std::string path;

    CudaHelper cuda_helper;

    sparse_matrix<float> sp_mat;
    sp_mat.rows = 10;
    sp_mat.columns = 9;
    sp_mat.nnz = 12;
    float values[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
    sp_mat.csr_val = values;
    int col_ind[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2};
    sp_mat.csr_col_ind = col_ind;
    int row_ptr[] = {0, 2, 2, 2, 5, 5, 6, 8, 8, 8, 10};
    sp_mat.csr_row_ptr = row_ptr;

    int start_row = 0;
    int end_row = 5;
    sparse_matrix<float> sp_mat_chunked = get_rows(&sp_mat, start_row, end_row);

    print_sparse_matrix(&sp_mat_chunked);

    return 1; // TODO
}


TEST_CASE("Sparse transpose", "[sparse][transpose]") {
    CHECK(test_sparse_transpose());
}

TEST_CASE("Sparse get rows", "[sparse][getrows]") {
    CHECK(test_get_rows());
}
