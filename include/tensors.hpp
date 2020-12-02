// Copyright 2020 Marcel Wagenländer

#ifndef TENSORS_H
#define TENSORS_H

#include <string>

#include "cuda_helper.hpp"


template<typename T>
class Matrix {
public:
    long num_rows_ = 0;
    long num_columns_ = 0;
    long size_ = 0;
    T *values_ = NULL;
    bool is_row_major_ = true;
    bool free_ = true;
    Matrix();
    Matrix(long num_rows, long num_columns, bool is_row_major);
    Matrix(long num_rows, long num_columns, bool is_row_major, bool free);
    Matrix(long num_rows, long num_columns, T *matrix_values, bool is_row_major, bool free);
    void set(long num_rows, long num_columns, bool is_row_major);
    void set(long num_rows, long num_columns, bool is_row_major, bool free);
    void set(long num_rows, long num_columns, T *matrix_values, bool is_row_major);
    void set(long num_rows, long num_columns, T *matrix_values, bool is_row_major, bool free);
    void set_random_values();
    void set_values(T value);
    ~Matrix();
};

template<typename T>
class SparseMatrix {
public:
    int num_rows_ = 0;
    int num_columns_ = 0;
    int nnz_ = 0;
    T *csr_val_ = NULL;
    int *csr_row_ptr_ = NULL;
    int *csr_col_ind_ = NULL;
    SparseMatrix();
    SparseMatrix(int num_rows, int num_columns, int num_nnz);
    SparseMatrix(int num_rows, int num_columns, int num_nnz, T *csr_val, int *csr_row_ptr, int *csr_col_ind);
    void set(int num_rows, int num_columns, int num_nnz);
    ~SparseMatrix();
};

template<typename T>
void print_matrix(Matrix<T> *mat);

template<typename T>
void print_matrix_features(Matrix<T> *mat);

template<typename T>
Matrix<T> load_npy_matrix(std::string path);

template<typename T>
SparseMatrix<T> load_mtx_matrix(std::string path);

template<typename T>
void save_npy_matrix(Matrix<T> *mat, std::string path);

template<typename T>
void save_npy_matrix_no_trans(Matrix<T> *mat, std::string path);

template<typename T>
void to_column_major(Matrix<T> *mat_col, Matrix<T> *mat);

template<typename T>
void to_row_major(Matrix<T> *mat_row, Matrix<T> *mat);

template<typename T>
void to_column_major_inplace(Matrix<T> *mat);

template<typename T>
void to_row_major_inplace(Matrix<T> *mat);

void get_rows(SparseMatrix<float> *reduced_mat, SparseMatrix<float> *mat, int start_row, int end_row);

void print_sparse_matrix(SparseMatrix<float> *mat);

void transpose_csr_matrix(SparseMatrix<float> *mat, CudaHelper *cuda_helper);

long count_nans(Matrix<float> *x);

bool check_nans(Matrix<float> *x, std::string name);

#endif
