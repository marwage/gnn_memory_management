// Copyright 2020 Marcel Wagenländer

#include "cuda_helper.hpp"
#include "sparse_computation.hpp"
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
    SparseMatrix<float> adjacency = load_mtx_matrix<float>(path);

    transpose_csr_matrix(&adjacency, &cuda_helper);

    return 1;// TODO
}

int test_sparse_transpose_cpu() {
    std::string home = std::getenv("HOME");
    std::string dir_path = home + "/gpu_memory_reduction/alzheimer/data";
    std::string flickr_dir_path = dir_path + "/flickr";
    std::string test_dir_path = dir_path + "/tests";
    std::string path;

    CudaHelper cuda_helper;

    path = flickr_dir_path + "/adjacency.mtx";
    SparseMatrix<float> adjacency = load_mtx_matrix<float>(path);

    transpose_csr_matrix_cpu(&adjacency);

    return 1;// TODO
}

int test_get_rows() {
    SparseMatrix<float> sp_mat;
    sp_mat.num_rows_ = 10;
    sp_mat.num_columns_ = 9;
    sp_mat.nnz_ = 12;
    float *values = new float[sp_mat.nnz_]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 11.0, 12.0, 9.0};
    sp_mat.csr_val_ = values;
    int *col_ind = new int[sp_mat.nnz_]{0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 8};
    sp_mat.csr_col_ind_ = col_ind;
    int *row_ptr = new int[sp_mat.num_rows_ + 1]{0, 2, 2, 2, 5, 5, 6, 8, 8, 8, 12};
    sp_mat.csr_row_ptr_ = row_ptr;

    int start_row = 2;
    int end_row = 7;
    SparseMatrix<float> sp_mat_reduced;
    get_rows(&sp_mat_reduced, &sp_mat, start_row, end_row);

    Matrix<float> mat;
    sparse_to_dense_matrix(&sp_mat, &mat);
    Matrix<float> mat_reduced(end_row + 1 - start_row, sp_mat.num_columns_, true);
    std::copy(mat.values_ + (start_row * mat_reduced.num_columns_),
              mat.values_ + (start_row * mat_reduced.num_columns_) + mat_reduced.size_,
              mat_reduced.values_);

    Matrix<float> test_mat_reduced;
    sparse_to_dense_matrix(&sp_mat_reduced, &test_mat_reduced);

    return check_equality(&test_mat_reduced, &mat_reduced);
}

int test_sparse_to_dense() {
    SparseMatrix<float> sp_mat;
    sp_mat.num_rows_ = 10;
    sp_mat.num_columns_ = 9;
    sp_mat.nnz_ = 12;
    float *values = new float[sp_mat.nnz_]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 11.0, 12.0, 9.0};
    sp_mat.csr_val_ = values;
    int *col_ind = new int[sp_mat.nnz_]{0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 8};
    sp_mat.csr_col_ind_ = col_ind;
    int *row_ptr = new int[sp_mat.num_rows_ + 1]{0, 2, 2, 2, 5, 5, 6, 8, 8, 8, 12};
    sp_mat.csr_row_ptr_ = row_ptr;

    Matrix<float> mat;

    sparse_to_dense_matrix(&sp_mat, &mat);

    print_matrix(&mat);

    return 1;// TODO
}

TEST_CASE("Sparse transpose", "[sparse][transpose]") {
    CHECK(test_sparse_transpose());
}

TEST_CASE("Sparse transpose, CPU", "[sparse][transpose][cpu]") {
    CHECK(test_sparse_transpose());
}

TEST_CASE("Sparse get rows", "[sparse][getrows]") {
    CHECK(test_get_rows());
}

TEST_CASE("Sparse matrix to dense matrix", "[sparse][todense]") {
    CHECK(test_sparse_to_dense());
}
