// Copyright 2020 Marcel Wagenländer

#ifndef TENSORS_H
#define TENSORS_H

#include <string>

#include "cuda_helper.hpp"


template<typename T>
struct matrix {
    long rows = 0;
    long columns = 0;
    T *values = 0;
    bool row_major = true;
};

template<typename T>
struct sparse_matrix {
    int rows = 0;
    int columns = 0;
    int nnz = 0;
    T *csr_val = NULL;
    int *csr_row_ptr = NULL;
    int *csr_col_ind = NULL;
};

template<typename T>
void print_matrix(matrix<T> *mat);

template<typename T>
void print_matrix_features(matrix<T> *mat);

template<typename T>
matrix<T> load_npy_matrix(std::string path);

template<typename T>
sparse_matrix<T> load_mtx_matrix(std::string path);

template<typename T>
void save_npy_matrix(matrix<T> *mat, std::string path);

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

sparse_matrix<float> get_rows(sparse_matrix<float> mat, int start_row, int end_row);

void print_sparse_matrix(sparse_matrix<float> mat);

matrix<float> new_float_matrix(long num_rows, long num_columns, bool row_major);

#endif
