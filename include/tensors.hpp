// Copyright 2020 Marcel Wagenländer

#ifndef TENSORS_H
#define TENSORS_H

#include <string>

#include "cuda_helper.hpp"


template<typename T>
struct matrix {
    long rows;
    long columns;
    T *values;
    bool row_major;
};

template<typename T>
struct sparse_matrix {
    int rows;
    int columns;
    int nnz;
    T *csr_val;
    int *csr_row_ptr;
    int *csr_col_ind;
};

template<typename T>
void print_matrix(matrix<T> mat);

template<typename T>
matrix<T> load_npy_matrix(std::string path);

template<typename T>
sparse_matrix<T> load_mtx_matrix(std::string path);

template<typename T>
void save_npy_matrix(matrix<T> mat, std::string path);

template<typename T>
void save_npy_matrix_no_trans(matrix<T> mat, std::string path);

template<typename T>
matrix<T> to_column_major(matrix<T> *mat);

template<typename T>
matrix<T> to_row_major(matrix<T> *mat);

template<typename T>
void to_column_major_inplace(matrix<T> *mat);

template<typename T>
void to_row_major_inplace(matrix<T> *mat);

matrix<float> add_matrices(CudaHelper *cuda_helper, matrix<float> mat_a, matrix<float> mat_b);

sparse_matrix<float> get_rows(sparse_matrix<float> mat, int start_row, int end_row);

void print_sparse_matrix(sparse_matrix<float> mat);

#endif
