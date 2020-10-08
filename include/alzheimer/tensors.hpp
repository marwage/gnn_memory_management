// Copyright 2020 Marcel Wagenländer

#ifndef TENSORS_H
#define TENSORS_H

#include <string>


template <typename T>
struct matrix {
    int rows;
    int columns;
    T *values;
};

template <typename T>
struct vector {
    int size;
    T *values;
};

template <typename T>
struct sparse_matrix {
    int rows;
    int columns;
    int nnz;
    T *csr_val;
    int *csr_row_ptr;
    int *csr_col_ind;
};

template <typename T>
void print_matrix(T* a, int rows, int cols);

template <typename T>
void print_vector(T* a, int num_ele);

template <typename T>
void transpose(T *a_T, T *a, int rows, int cols);

template <typename T>
void one_to_zero_index(T *a, int len);

template <typename T>
vector<T> load_npy_vector(std::string path);

template <typename T>
matrix<T> load_npy_matrix(std::string path);

template <typename T>
sparse_matrix<T> load_mtx_matrix(std::string path);

template <typename T>
void save_npy_matrix(matrix<T> mat, std::string path);

#endif
