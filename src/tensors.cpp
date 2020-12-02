// Copyright 2020 Marcel Wagenländer

#include "tensors.hpp"

#include "cnpy.h"
#include "mmio_wrapper.hpp"

#include <cstring>
#include <iostream>
#include <thread>


template<typename T>
Matrix<T>::Matrix() {}
template Matrix<float>::Matrix();
template Matrix<int>::Matrix();

template<typename T>
Matrix<T>::Matrix(long num_rows, long num_columns, bool is_row_major) {
    set(num_rows, num_columns, is_row_major);
}
template Matrix<int>::Matrix(long num_rows, long num_columns, bool is_row_major);
template Matrix<float>::Matrix(long num_rows, long num_columns, bool is_row_major);

template<typename T>
Matrix<T>::Matrix(long num_rows, long num_columns, bool is_row_major, bool free) {
    set(num_rows, num_columns, is_row_major, free);
}
template Matrix<int>::Matrix(long num_rows, long num_columns, bool is_row_major, bool free);
template Matrix<float>::Matrix(long num_rows, long num_columns, bool is_row_major, bool free);

template<typename T>
Matrix<T>::Matrix(long num_rows, long num_columns, T *matrix_values, bool is_row_major, bool free) {
    set(num_rows, num_columns, matrix_values, is_row_major, free);
}
template Matrix<float>::Matrix(long num_rows, long num_columns, float *matrix_values, bool is_row_major, bool free);
template Matrix<int>::Matrix(long num_rows, long num_columns, int *matrix_values, bool is_row_major, bool free);

template<typename T>
Matrix<T>::~Matrix() {
    if (free_) {
        delete[] values_;
    }
}
template Matrix<float>::~Matrix();
template Matrix<int>::~Matrix();

template<typename T>
void Matrix<T>::set(long num_rows, long num_columns, bool is_row_major) {
    set(num_rows, num_columns, is_row_major, true);
}
template void Matrix<int>::set(long num_rows, long num_columns, bool is_row_major);
template void Matrix<float>::set(long num_rows, long num_columns, bool is_row_major);

template<typename T>
void Matrix<T>::set(long num_rows, long num_columns, bool is_row_major, bool free) {
    num_rows_ = num_rows;
    num_columns_ = num_columns;
    size_ = num_rows_ * num_columns;
    if (values_ != NULL && free_) {
        delete[] values_;
    }
    values_ = new T[size_];
    is_row_major_ = is_row_major;
    free_ = free;
}
template void Matrix<int>::set(long num_rows, long num_columns, bool is_row_major, bool free);
template void Matrix<float>::set(long num_rows, long num_columns, bool is_row_major, bool free);

template<typename T>
void Matrix<T>::set(long num_rows, long num_columns, T *matrix_values, bool is_row_major) {
    num_rows_ = num_rows;
    num_columns_ = num_columns;
    size_ = num_rows_ * num_columns;
    values_ = matrix_values;
    is_row_major_ = is_row_major;
    free_ = true;
}
template void Matrix<float>::set(long num_rows, long num_columns, float *matrix_values, bool is_row_major);
template void Matrix<int>::set(long num_rows, long num_columns, int *matrix_values, bool is_row_major);

template<typename T>
void Matrix<T>::set(long num_rows, long num_columns, T *matrix_values, bool is_row_major, bool free) {
    num_rows_ = num_rows;
    num_columns_ = num_columns;
    size_ = num_rows_ * num_columns;
    values_ = matrix_values;
    is_row_major_ = is_row_major;
    free_ = free;
}
template void Matrix<float>::set(long num_rows, long num_columns, float *matrix_values, bool is_row_major, bool free);
template void Matrix<int>::set(long num_rows, long num_columns, int *matrix_values, bool is_row_major, bool free);

template<typename T>
void Matrix<T>::set_random_values() {
    for (long i = 0; i < num_rows_ * num_columns_; ++i) {
        values_[i] = rand();
    }
}
template void Matrix<float>::set_random_values();
template void Matrix<int>::set_random_values();

template<typename T>
void Matrix<T>::set_values(T value) {
    for (long i = 0; i < num_rows_ * num_columns_; ++i) {
        values_[i] = value;
    }
}
template void Matrix<float>::set_values(float value);
template void Matrix<int>::set_values(int value);

template<typename T>
SparseMatrix<T>::SparseMatrix() {}
template SparseMatrix<float>::SparseMatrix();

template<typename T>
SparseMatrix<T>::SparseMatrix(int num_rows, int num_columns, int num_nnz) {
    set(num_rows, num_columns, num_nnz);
}
template SparseMatrix<float>::SparseMatrix(int num_rows, int num_columns, int num_nnz);

template<typename T>
SparseMatrix<T>::SparseMatrix(int num_rows, int num_columns, int num_nnz, T *csr_val, int *csr_row_ptr, int *csr_col_ind) {
    num_rows_ = num_rows;
    num_columns_ = num_columns;
    nnz_ = num_nnz;
    csr_val_ = csr_val;
    csr_row_ptr_ = csr_row_ptr;
    csr_col_ind_ = csr_col_ind;
}
template SparseMatrix<float>::SparseMatrix(int num_rows, int num_columns, int num_nnz, float *csr_val, int *csr_row_ptr, int *csr_col_ind);

template<typename T>
SparseMatrix<T>::~SparseMatrix() {
    delete[] csr_col_ind_;
    delete[] csr_row_ptr_;
    delete[] csr_val_;
}
template SparseMatrix<float>::~SparseMatrix();

template<typename T>
void SparseMatrix<T>::set(int num_rows, int num_columns, int num_nnz) {
    num_rows_ = num_rows;
    num_columns_ = num_columns;
    nnz_ = num_nnz;
    csr_val_ = new T[nnz_];
    csr_row_ptr_ = new int[num_rows_ + 1];
    csr_col_ind_ = new int[nnz_];
}
template void SparseMatrix<float>::set(int num_rows, int num_columns, int num_nnz);

template<typename T>
void print_matrix(Matrix<T> *mat) {
    int N;
    if (mat->num_rows_ < 10) {
        N = mat->num_rows_;
    } else {
        N = 10;
    }
    int M;
    if (mat->num_columns_ < 10) {
        M = mat->num_columns_;
    } else {
        M = 10;
    }

    std::cout << "-----" << std::endl;
    for (int i = 0; i < N; i = i + 1) {
        for (int j = 0; j < M; j = j + 1) {
            if (mat->is_row_major_) {
                std::cout << mat->values_[i * mat->num_columns_ + j] << ",";
            } else {
                std::cout << mat->values_[j * mat->num_rows_ + i] << ",";
            }
        }
        std::cout << std::endl;
    }
}
template void print_matrix<float>(Matrix<float> *mat);
template void print_matrix<int>(Matrix<int> *mat);

template<typename T>
void print_matrix_features(Matrix<T> *mat) {
    std::cout << "Shape: (" << mat->num_rows_ << ", " << mat->num_columns_ << ")" << std::endl;
    std::cout << "Row major: " << mat->is_row_major_ << std::endl;
    std::cout << "Values pointer: " << mat->values_ << std::endl;
}
template void print_matrix_features<float>(Matrix<float> *mat);
template void print_matrix_features<int>(Matrix<int> *mat);


long new_index(long old_idx, long rows, long cols) {
    long last_idx = rows * cols - 1;
    if (old_idx == last_idx) {
        return last_idx;
    } else {
        long new_idx = rows * old_idx;
        new_idx = new_idx % last_idx;
        return new_idx;
    }
}

template<typename T>
void transpose(T *a_T, T *a, long rows, long cols,
               long rows_lower, long rows_upper, long columns_lower, long columns_upper) {
    if (rows_upper - rows_lower < 1 ||
        columns_upper - columns_lower < 1) {
        throw "Wrong boundaries";
    }
    int boundary = 4096;
    if (rows_upper - rows_lower < boundary) {
        if (columns_upper - columns_lower < boundary) {

            long old_idx, new_idx;
            for (long i = rows_lower; i < rows_upper; ++i) {
                for (long j = columns_lower; j < columns_upper; ++j) {
                    old_idx = i * cols + j;

                    new_idx = new_index(old_idx, rows, cols);
                    a_T[new_idx] = a[old_idx];
                }
            }
        } else {
            int column_mid = (columns_upper - columns_lower) / 2 + columns_lower;
            std::thread thread_one(transpose<T>, a_T, a, rows, cols, rows_lower, rows_upper, columns_lower, column_mid);
            std::thread thread_two(transpose<T>, a_T, a, rows, cols, rows_lower, rows_upper, column_mid, columns_upper);
            thread_one.join();
            thread_two.join();
        }
    } else {
        if (columns_upper - columns_lower < boundary) {
            int row_mid = (rows_upper - rows_lower) / 2 + rows_lower;
            std::thread thread_one(transpose<T>, a_T, a, rows, cols, rows_lower, row_mid, columns_lower, columns_upper);
            std::thread thread_two(transpose<T>, a_T, a, rows, cols, row_mid, rows_upper, columns_lower, columns_upper);
            thread_one.join();
            thread_two.join();
        } else {
            int row_mid = (rows_upper - rows_lower) / 2 + rows_lower;
            int column_mid = (columns_upper - columns_lower) / 2 + columns_lower;
            std::thread thread_one(transpose<T>, a_T, a, rows, cols, rows_lower, row_mid, columns_lower, column_mid);
            std::thread thread_two(transpose<T>, a_T, a, rows, cols, rows_lower, row_mid, column_mid, columns_upper);
            std::thread thread_three(transpose<T>, a_T, a, rows, cols, row_mid, rows_upper, columns_lower, column_mid);
            std::thread thread_four(transpose<T>, a_T, a, rows, cols, row_mid, rows_upper, column_mid, columns_upper);
            thread_one.join();
            thread_two.join();
            thread_three.join();
            thread_four.join();
        }
    }
}

template<typename T>
void *transpose(T *a_T, T *a, long rows, long cols) {
    transpose(a_T, a, rows, cols, 0, rows, 0, cols);
}

template<typename T>
void one_to_zero_index(T *a, int len) {
    for (int i = 0; i < len; ++i) {
        a[i] = a[i] - 1;
    }
}

template void one_to_zero_index<int>(int *a, int len);


template<typename T>
Matrix<T> load_npy_matrix(std::string path) {
    cnpy::NpyArray arr = cnpy::npy_load(path);
    if (arr.word_size != sizeof(T)) {
        std::cout << "Matrix has wrong data type" << std::endl;
    }
    T *arr_data = arr.data<T>();
    long num_columns;
    if (arr.shape.size() == 1) {
        num_columns = 1;
    } else {
        num_columns = arr.shape[1];
    }
    Matrix<T> mat(arr.shape[0], num_columns, true);
    std::memcpy(mat.values_, arr_data, mat.num_rows_ * mat.num_columns_ * sizeof(T));

    return mat;
}

template Matrix<float> load_npy_matrix<float>(std::string path);
template Matrix<int> load_npy_matrix<int>(std::string path);
template Matrix<bool> load_npy_matrix<bool>(std::string path);


template<typename T>
SparseMatrix<T> load_mtx_matrix(std::string path) {
    char *path_char = &*path.begin();
    SparseMatrix<T> sp_mat;
    int err = loadMMSparseMatrix<T>(path_char, 'f', true,
                                    &sp_mat.num_rows_, &sp_mat.num_columns_, &sp_mat.nnz_,
                                    &sp_mat.csr_val_, &sp_mat.csr_row_ptr_,
                                    &sp_mat.csr_col_ind_, true);
    if (err) {
        std::cout << "loadMMSparseMatrix failed" << std::endl;
    }
    one_to_zero_index(sp_mat.csr_row_ptr_, sp_mat.num_rows_ + 1);
    one_to_zero_index(sp_mat.csr_col_ind_, sp_mat.nnz_);

    return sp_mat;
}

template SparseMatrix<float> load_mtx_matrix<float>(std::string path);

template<typename T>
void save_npy_matrix(Matrix<T> *mat, std::string path) {
    to_row_major_inplace(mat);
    std::vector<size_t> shape = {(size_t) mat->num_rows_, (size_t) mat->num_columns_};
    cnpy::npy_save<T>(path, mat->values_, shape);
}

template void save_npy_matrix<float>(Matrix<float> *mat, std::string path);
template void save_npy_matrix<int>(Matrix<int> *mat, std::string path);

template<typename T>
void save_npy_matrix_no_trans(Matrix<T> *mat, std::string path) {
    std::vector<size_t> shape = {(size_t) mat->num_rows_, (size_t) mat->num_columns_};
    cnpy::npy_save<T>(path, mat->values_, shape);
}

template void save_npy_matrix_no_trans<float>(Matrix<float> *mat, std::string path);
template void save_npy_matrix_no_trans<int>(Matrix<int> *mat, std::string path);

template<typename T>
void to_column_major_inplace(Matrix<T> *mat) {
    if (mat->is_row_major_) {
        T *values_T = new T[mat->size_];
        transpose<T>(values_T, mat->values_, mat->num_rows_, mat->num_columns_);
        if (mat->free_) {
            delete[] mat->values_;
        }
        mat->values_ = values_T;
        mat->is_row_major_ = false;
    }
}

template void to_column_major_inplace<float>(Matrix<float> *mat);
template void to_column_major_inplace<int>(Matrix<int> *mat);

template<typename T>
void to_column_major(Matrix<T> *mat_col, Matrix<T> *mat) {
    if (mat->is_row_major_) {
        T *a_T = new T[mat->size_];
        transpose<T>(a_T, mat->values_, mat->num_rows_, mat->num_columns_);
        mat_col->set(mat->num_rows_, mat->num_columns_, a_T, false);
    } else {
        mat_col = mat;
    }
}
template void to_column_major<float>(Matrix<float> *mat_col, Matrix<float> *mat);
template void to_column_major<int>(Matrix<int> *mat_col, Matrix<int> *mat);


template<typename T>
void to_row_major_inplace(Matrix<T> *mat) {
    if (!mat->is_row_major_) {
        T *values_T = new T[mat->size_];
        transpose<T>(values_T, mat->values_, mat->num_columns_, mat->num_rows_);
        if (mat->free_) {
            delete[] mat->values_;
        }
        mat->values_ = values_T;
        mat->is_row_major_ = true;
    }
}
template void to_row_major_inplace<float>(Matrix<float> *mat);
template void to_row_major_inplace<int>(Matrix<int> *mat);

template<typename T>
void to_row_major(Matrix<T> *mat_row, Matrix<T> *mat) {
    if (!mat->is_row_major_) {
        T *a_T = new T[mat->size_];
        transpose<T>(a_T, mat->values_, mat->num_columns_, mat->num_rows_);
        mat_row->set(mat->num_rows_, mat->num_columns_, a_T, true);
    } else {
        mat_row = mat;
    }
}
template void to_row_major<float>(Matrix<float> *mat_row, Matrix<float> *mat);
template void to_row_major<int>(Matrix<int> *mat_row, Matrix<int> *mat);

void get_rows(SparseMatrix<float> *reduced_mat, SparseMatrix<float> *mat, int start_row, int end_row) {
    int first_index = mat->csr_row_ptr_[start_row];
    int last_index = mat->csr_row_ptr_[end_row];

    reduced_mat->set(end_row - start_row, mat->num_columns_, last_index - first_index);

    std::memcpy(reduced_mat->csr_val_, &mat->csr_val_[first_index], reduced_mat->nnz_ * sizeof(float));
    std::memcpy(reduced_mat->csr_row_ptr_, &mat->csr_row_ptr_[start_row], (reduced_mat->num_rows_ + 1) * sizeof(int));
    std::memcpy(reduced_mat->csr_col_ind_, &mat->csr_col_ind_[first_index], reduced_mat->nnz_ * sizeof(int));
    for (int i = 0; i < reduced_mat->num_rows_; ++i) {
        reduced_mat->csr_row_ptr_[i] = reduced_mat->csr_row_ptr_[i] - first_index;
    }
}

void print_sparse_matrix(SparseMatrix<float> *mat) {
    std::cout << "Row pointers" << std::endl;
    for (int i = 0; i < mat->num_rows_ + 1; ++i) {
        std::cout << mat->csr_row_ptr_[i] << ", ";
    }
    std::cout << std::endl;
    std::cout << "Column indices" << std::endl;
    for (int i = 0; i < mat->nnz_; ++i) {
        std::cout << mat->csr_col_ind_[i] << ", ";
    }
    std::cout << std::endl;
    std::cout << "Values" << std::endl;
    for (int i = 0; i < mat->nnz_; ++i) {
        std::cout << mat->csr_val_[i] << ", ";
    }
    std::cout << std::endl;
}

void transpose_csr_matrix(SparseMatrix<float> *mat, CudaHelper *cuda_helper) {
    float *d_mat_csr_val;
    int *d_mat_csr_row_ptr, *d_mat_csr_col_ind;
    check_cuda(cudaMalloc(&d_mat_csr_val, mat->nnz_ * sizeof(float)));
    check_cuda(cudaMalloc(&d_mat_csr_row_ptr, (mat->num_rows_ + 1) * sizeof(int)));
    check_cuda(cudaMalloc(&d_mat_csr_col_ind, mat->nnz_ * sizeof(int)));
    check_cuda(cudaMemcpy(d_mat_csr_val, mat->csr_val_,
                          mat->nnz_ * sizeof(float), cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(d_mat_csr_row_ptr, mat->csr_row_ptr_,
                          (mat->num_rows_ + 1) * sizeof(int), cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(d_mat_csr_col_ind, mat->csr_col_ind_,
                          mat->nnz_ * sizeof(int), cudaMemcpyHostToDevice));

    float *d_mat_csc_val;
    int *d_mat_csc_col_ptr, *d_mat_csc_row_ind;
    check_cuda(cudaMalloc(&d_mat_csc_val, mat->nnz_ * sizeof(float)));
    check_cuda(cudaMalloc(&d_mat_csc_col_ptr, (mat->num_columns_ + 1) * sizeof(int)));
    check_cuda(cudaMalloc(&d_mat_csc_row_ind, mat->nnz_ * sizeof(int)));

    size_t buffer_size;
    check_cusparse(cusparseCsr2cscEx2_bufferSize(cuda_helper->cusparse_handle,
                                                 mat->num_rows_, mat->num_columns_, mat->nnz_,
                                                 d_mat_csr_val, d_mat_csr_row_ptr, d_mat_csr_col_ind,
                                                 d_mat_csc_val, d_mat_csc_col_ptr, d_mat_csc_row_ind,
                                                 CUDA_R_32F,
                                                 CUSPARSE_ACTION_SYMBOLIC,// could try CUSPARSE_ACTION_NUMERIC
                                                 CUSPARSE_INDEX_BASE_ZERO,
                                                 CUSPARSE_CSR2CSC_ALG1,// could try CUSPARSE_CSR2CSC_ALG2
                                                 &buffer_size));

    void *d_buffer;
    check_cuda(cudaMalloc(&d_buffer, buffer_size));

    check_cusparse(cusparseCsr2cscEx2(cuda_helper->cusparse_handle,
                                      mat->num_rows_, mat->num_columns_, mat->nnz_,
                                      d_mat_csr_val, d_mat_csr_row_ptr, d_mat_csr_col_ind,
                                      d_mat_csc_val, d_mat_csc_col_ptr, d_mat_csc_row_ind,
                                      CUDA_R_32F,
                                      CUSPARSE_ACTION_SYMBOLIC,// could try CUSPARSE_ACTION_NUMERIC
                                      CUSPARSE_INDEX_BASE_ZERO,
                                      CUSPARSE_CSR2CSC_ALG1,// could try CUSPARSE_CSR2CSC_ALG2
                                      d_buffer));

    long tmp = mat->num_rows_;
    mat->num_rows_ = mat->num_columns_;
    mat->num_columns_ = tmp;
    delete[] mat->csr_row_ptr_;
    delete[] mat->csr_col_ind_;
    mat->csr_row_ptr_ = new int[mat->num_rows_ + 1];
    mat->csr_col_ind_ = new int[mat->nnz_];

    check_cuda(cudaMemcpy(mat->csr_row_ptr_, d_mat_csc_col_ptr,
                          (mat->num_rows_ + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    check_cuda(cudaMemcpy(mat->csr_col_ind_, d_mat_csc_row_ind,
                          mat->nnz_ * sizeof(int), cudaMemcpyDeviceToHost));

    // free GPU memory
    check_cuda(cudaFree(d_buffer));
    check_cuda(cudaFree(d_mat_csc_row_ind));
    check_cuda(cudaFree(d_mat_csc_col_ptr));
    check_cuda(cudaFree(d_mat_csc_val));
    check_cuda(cudaFree(d_mat_csr_col_ind));
    check_cuda(cudaFree(d_mat_csr_row_ptr));
    check_cuda(cudaFree(d_mat_csr_val));
}

long count_nans(Matrix<float> *x) {
    long num_nans = 0;

    for (int i = 0; i < x->num_rows_ * x->num_columns_; ++i) {
        if (isnan(x->values_[i])) {
            num_nans = num_nans + 1;
        }
    }

    return num_nans;
}

bool check_nans(Matrix<float> *x, std::string name) {
    long num_nans = count_nans(x);
    if (num_nans > 0) {
        std::cout << name << " has " << num_nans << " NaNs" << std::endl;
        return true;
    } else {
        return false;
    }
}
