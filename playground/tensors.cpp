// Copyright 2020 Marcel Wagenländer

#include "tensors.hpp"

#include <string>
#include <iostream>
#include <thread>


template<typename T>
Matrix<T>::Matrix() {}
template Matrix<float>::Matrix();


template<typename T>
Matrix<T>::Matrix(long num_rows, long num_columns, bool row_major) {
    set(num_rows, num_columns, row_major);
}
template Matrix<int>::Matrix(long num_rows, long num_columns, bool row_major);
template Matrix<float>::Matrix(long num_rows, long num_columns, bool row_major);

template<typename T>
Matrix<T>::Matrix(long num_rows, long num_columns, T *values, bool row_major) {
    num_rows_ = num_rows;
    num_columns_ = num_columns;
    size_ = num_rows_ * num_columns_;
    values_ = values;
    row_major_ = row_major;
}
template Matrix<float>::Matrix(long num_rows, long num_columns, float *values, bool row_major);
template Matrix<int>::Matrix(long num_rows, long num_columns, int *values, bool row_major);

template<typename T>
Matrix<T>::~Matrix() {
    std::cout << "Matrix deconstructor called" << std::endl;
    delete[] values_;
}
template Matrix<float>::~Matrix();
template Matrix<int>::~Matrix();

template<typename T>
void Matrix<T>::set(long num_rows, long num_columns, bool row_major) {
    num_rows_ = num_rows;
    num_columns_ = num_columns;
    size_ = num_rows_ * num_columns_;
    row_major_ = row_major;
    values_ = new T[size_]();
}
template void Matrix<float>::set(long num_rows, long num_columns, bool row_major);
template void Matrix<int>::set(long num_rows, long num_columns, bool row_major);

template<typename T>
SparseMatrix<T>::SparseMatrix() {}
template SparseMatrix<float>::SparseMatrix();

template<typename T>
SparseMatrix<T>::SparseMatrix(int num_rows, int num_columns, int num_nnz) {
    rows = num_rows;
    columns = num_columns;
    nnz = num_nnz;
    csr_val = new T[nnz]();
    csr_row_ptr = new int[rows + 1]();
    csr_col_ind = new int[nnz]();
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
    delete[] csr_col_ind;
    delete[] csr_row_ptr;
    delete[] csr_val;
}
template SparseMatrix<float>::~SparseMatrix();
