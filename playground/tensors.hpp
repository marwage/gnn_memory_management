// Copyright 2020 Marcel Wagenländer

#ifndef TENSORS_H
#define TENSORS_H

#include <string>


template<typename T>
class Matrix {
public:
    long rows = 0;
    long columns = 0;
    long size_ = 0;
    T *values = NULL;
    bool row_major = true;
    Matrix();
    Matrix(long num_rows, long num_columns, bool is_row_major);
    Matrix(long num_rows, long num_columns, T *matrix_values, bool is_row_major);
    ~Matrix();
};

template<typename T>
class SparseMatrix {
public:
    int rows = 0;
    int columns = 0;
    int nnz = 0;
    T *csr_val = NULL;
    int *csr_row_ptr = NULL;
    int *csr_col_ind = NULL;
    SparseMatrix();
    SparseMatrix(int num_rows, int num_columns, int num_nnz);
    SparseMatrix(int num_rows, int num_columns, int num_nnz, T *values, int *row_ptr, int *col_ind);
    ~SparseMatrix();
};

#endif
