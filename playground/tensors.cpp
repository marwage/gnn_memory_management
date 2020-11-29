// Copyright 2020 Marcel Wagenländer

#include "tensors.hpp"

#include "cnpy.h"
#include "mmio_wrapper.hpp"

#include <cstring>
#include <iostream>
#include <thread>

#include "catch2/catch.hpp" // DEBUGGING


template<typename T>
Matrix<T>::Matrix() {}
template Matrix<int>::Matrix();
template Matrix<float>::Matrix();

template<typename T>
Matrix<T>::Matrix(long num_rows, long num_columns, bool is_row_major) {
    rows = num_rows;
    columns = num_columns;
    size_ = rows * columns;
    values = new T[size_];
    row_major = is_row_major;

    for (int i = 0; i < size_; ++i) {
        values[i] = 0;
    }
    T *ptr = values;
    int num = 0;
}
template Matrix<int>::Matrix(long num_rows, long num_columns, bool is_row_major);
template Matrix<float>::Matrix(long num_rows, long num_columns, bool is_row_major);

template<typename T>
Matrix<T>::Matrix(long num_rows, long num_columns, T *matrix_values, bool is_row_major) {
    rows = num_rows;
    columns = num_columns;
    size_ = rows * columns;
    values = matrix_values;
    row_major = is_row_major;
}
template Matrix<float>::Matrix(long num_rows, long num_columns, float *matrix_values, bool is_row_major);
template Matrix<int>::Matrix(long num_rows, long num_columns, int *matrix_values, bool is_row_major);

template<typename T>
Matrix<T>::~Matrix() {
    delete[] values;
}
template Matrix<float>::~Matrix();
template Matrix<int>::~Matrix();

template<typename T>
SparseMatrix<T>::SparseMatrix() {}
template SparseMatrix<float>::SparseMatrix();

template<typename T>
SparseMatrix<T>::SparseMatrix(int num_rows, int num_columns, int num_nnz) {
    rows = num_rows;
    columns = num_columns;
    nnz = num_nnz;
    csr_val = new T[nnz];
    csr_row_ptr = new int[rows + 1];
    csr_col_ind = new int[nnz];
}
template SparseMatrix<float>::SparseMatrix(int num_rows, int num_columns, int num_nnz);

template<typename T>
SparseMatrix<T>::SparseMatrix(int num_rows, int num_columns, int num_nnz, T *values, int *row_ptr, int *col_ind) {
    rows = num_rows;
    columns = num_columns;
    nnz = num_nnz;
    csr_val = values;
    csr_row_ptr = row_ptr;
    csr_col_ind = col_ind;
}
template SparseMatrix<float>::SparseMatrix(int num_rows, int num_columns, int num_nnz, float *values, int *row_ptr, int *col_ind);

template<typename T>
SparseMatrix<T>::~SparseMatrix() {
    delete csr_col_ind;
    delete csr_row_ptr;
    delete csr_val;
}
template SparseMatrix<float>::~SparseMatrix();
