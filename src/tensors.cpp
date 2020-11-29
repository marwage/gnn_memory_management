// Copyright 2020 Marcel Wagenländer

#include "tensors.hpp"

#include "cnpy.h"
#include "mmio_wrapper.hpp"

#include <cstring>
#include <iostream>
#include <thread>


template<typename T>
matrix<T>::matrix() {}

template<typename T>
matrix<T>::~matrix() {
    delete values;
}

template<typename T>
sparse_matrix<T>::sparse_matrix() {}

template<typename T>
sparse_matrix<T>::~sparse_matrix() {
    delete csr_col_ind;
    delete csr_row_ptr;
    delete csr_val;
}

template<typename T>
void print_matrix(matrix<T> *mat) {
    int N;
    if (mat->rows < 10) {
        N = mat->rows;
    } else {
        N = 10;
    }
    int M;
    if (mat->columns < 10) {
        M = mat->columns;
    } else {
        M = 10;
    }

    std::cout << "-----" << std::endl;
    for (int i = 0; i < N; i = i + 1) {
        for (int j = 0; j < M; j = j + 1) {
            if (mat->row_major) {
                std::cout << mat->values[i * mat->columns + j] << ",";
            } else {
                std::cout << mat->values[j * mat->rows + i] << ",";
            }
        }
        std::cout << std::endl;
    }
}
template void print_matrix<float>(matrix<float> *mat);
template void print_matrix<int>(matrix<int> *mat);

template<typename T>
void print_matrix_features(matrix<T> *mat) {
    std::cout << "Shape: (" << mat->rows << ", " << mat->columns << ")" << std::endl;
    std::cout << "Row major: " << mat->row_major << std::endl;
    std::cout << "Values pointer: " << mat->values << std::endl;
}
template void print_matrix_features<float>(matrix<float> *mat);
template void print_matrix_features<int>(matrix<int> *mat);


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
T* transpose(T *a, long rows, long cols) {
    T *a_T = new T[rows * cols];

    T first_value = a[0]; // DEBUGGING

    transpose(a_T, a, rows, cols, 0, rows, 0, cols);

    return a_T;
}

template<typename T>
void one_to_zero_index(T *a, int len) {
    for (int i = 0; i < len; ++i) {
        a[i] = a[i] - 1;
    }
}

template void one_to_zero_index<int>(int *a, int len);


template<typename T>
matrix<T> load_npy_matrix(std::string path) {
    cnpy::NpyArray arr = cnpy::npy_load(path);
    if (arr.word_size != sizeof(T)) {
        std::cout << "Matrix has wrong data type" << std::endl;
    }
    T *arr_data = arr.data<T>();
    matrix<T> mat;
    mat.rows = arr.shape[0];
    if (arr.shape.size() == 1) {
        mat.columns = 1;
    } else {
        mat.columns = arr.shape[1];
    }
    mat.values = new T[mat.rows * mat.columns];
    mat.row_major = true;
    std::memcpy(mat.values, arr_data, mat.rows * mat.columns * sizeof(T));

    return mat;
}

template matrix<float> load_npy_matrix<float>(std::string path);
template matrix<int> load_npy_matrix<int>(std::string path);
template matrix<bool> load_npy_matrix<bool>(std::string path);


template<typename T>
sparse_matrix<T> load_mtx_matrix(std::string path) {
    char *path_char = &*path.begin();
    sparse_matrix<T> sp_mat;
    int err = loadMMSparseMatrix<T>(path_char, 'f', true,
                                    &sp_mat.rows, &sp_mat.columns, &sp_mat.nnz,
                                    &sp_mat.csr_val, &sp_mat.csr_row_ptr,
                                    &sp_mat.csr_col_ind, true);
    if (err) {
        std::cout << "loadMMSparseMatrix failed" << std::endl;
    }
    one_to_zero_index(sp_mat.csr_row_ptr, sp_mat.rows + 1);
    one_to_zero_index(sp_mat.csr_col_ind, sp_mat.nnz);

    return sp_mat;
}

template sparse_matrix<float> load_mtx_matrix<float>(std::string path);

template<typename T>
void save_npy_matrix(matrix<T> *mat, std::string path) {
    to_row_major_inplace(mat);
    std::vector<size_t> shape = {(size_t) mat->rows, (size_t) mat->columns};
    cnpy::npy_save<T>(path, mat->values, shape);
}

template void save_npy_matrix<float>(matrix<float> *mat, std::string path);
template void save_npy_matrix<int>(matrix<int> *mat, std::string path);

template<typename T>
void save_npy_matrix(matrix<T> mat, std::string path) {
    save_npy_matrix(&mat, path);
}

template void save_npy_matrix<float>(matrix<float> mat, std::string path);
template void save_npy_matrix<int>(matrix<int> mat, std::string path);

template<typename T>
void save_npy_matrix_no_trans(matrix<T> mat, std::string path) {
    std::vector<size_t> shape = {(size_t) mat.rows, (size_t) mat.columns};
    cnpy::npy_save<T>(path, mat.values, shape);
}

template void save_npy_matrix_no_trans<float>(matrix<float> mat, std::string path);
template void save_npy_matrix_no_trans<int>(matrix<int> mat, std::string path);

template<typename T>
void to_column_major_inplace(matrix<T> *mat) {
    if (mat->row_major) {
        T* values_T = transpose<T>(mat->values, mat->rows, mat->columns);
        delete mat->values;
        mat->values = values_T;
        mat->row_major = false;
    }
}

template void to_column_major_inplace<float>(matrix<float> *mat);
template void to_column_major_inplace<int>(matrix<int> *mat);

template<typename T>
matrix<T> to_column_major(matrix<T> *mat) {
    throw "Not implemented";
}
template matrix<float> to_column_major<float>(matrix<float> *mat);
template matrix<int> to_column_major<int>(matrix<int> *mat);


template<typename T>
void to_row_major_inplace(matrix<T> *mat) {
    if (!mat->row_major) {
        T* values_T = transpose<T>(mat->values, mat->columns, mat->rows);
        delete mat->values;
        mat->values = values_T;
        mat->row_major = true;
    }
}
template void to_row_major_inplace<float>(matrix<float> *mat);
template void to_row_major_inplace<int>(matrix<int> *mat);

template<typename T>
matrix<T> to_row_major(matrix<T> *mat) {
    throw "Not implemented";
}
template matrix<float> to_row_major<float>(matrix<float> *mat);
template matrix<int> to_row_major<int>(matrix<int> *mat);

sparse_matrix<float> get_rows(sparse_matrix<float> *mat, int start_row, int end_row) {
    sparse_matrix<float> reduced_mat;
    reduced_mat.rows = end_row - start_row;
    reduced_mat.columns = mat->columns;

    int first_index = mat->csr_row_ptr[start_row];
    int last_index = mat->csr_row_ptr[end_row];
    reduced_mat.nnz = last_index - first_index;
    reduced_mat.csr_val = new float[reduced_mat.nnz];
    reduced_mat.csr_row_ptr = new int[(reduced_mat.rows + 1)];
    reduced_mat.csr_col_ind = new int[reduced_mat.nnz];

    std::memcpy(reduced_mat.csr_val, &mat->csr_val[first_index], reduced_mat.nnz * sizeof(float));
    std::memcpy(reduced_mat.csr_row_ptr, &mat->csr_row_ptr[start_row], (reduced_mat.rows + 1) * sizeof(int));
    std::memcpy(reduced_mat.csr_col_ind, &mat->csr_col_ind[first_index], reduced_mat.nnz * sizeof(int));
    for (int i = 0; i < reduced_mat.rows; ++i) {
        reduced_mat.csr_row_ptr[i] = reduced_mat.csr_row_ptr[i] - first_index;
    }

    return reduced_mat;
}

void print_sparse_matrix(sparse_matrix<float> *mat) {
    std::cout << "Row pointers" << std::endl;
    for (int i = 0; i < mat->rows + 1; ++i) {
        std::cout << mat->csr_row_ptr[i] << ", ";
    }
    std::cout << std::endl;
    std::cout << "Column indices" << std::endl;
    for (int i = 0; i < mat->nnz; ++i) {
        std::cout << mat->csr_col_ind[i] << ", ";
    }
    std::cout << std::endl;
    std::cout << "Values" << std::endl;
    for (int i = 0; i < mat->nnz; ++i) {
        std::cout << mat->csr_val[i] << ", ";
    }
    std::cout << std::endl;
}

matrix<float> new_float_matrix(long num_rows, long num_columns, bool row_major) {
    matrix<float> mat;
    mat.rows = num_rows;
    mat.columns = num_columns;
    mat.values = new float[mat.rows * mat.columns];
    if (row_major) {
        mat.row_major = true;
    } else {
        mat.row_major = false;
    }

    return mat;
}

void transpose_csr_matrix(sparse_matrix<float> *mat, CudaHelper *cuda_helper){
    float *d_mat_csr_val;
    int *d_mat_csr_row_ptr, *d_mat_csr_col_ind;
    check_cuda(cudaMalloc(&d_mat_csr_val,mat->nnz * sizeof(float)));
    check_cuda(cudaMalloc(&d_mat_csr_row_ptr,(mat->rows + 1) * sizeof(int)));
    check_cuda(cudaMalloc(&d_mat_csr_col_ind,mat->nnz * sizeof(int)));
    check_cuda(cudaMemcpy(d_mat_csr_val, mat->csr_val,
                          mat->nnz * sizeof(float), cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(d_mat_csr_row_ptr, mat->csr_row_ptr,
                          (mat->rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(d_mat_csr_col_ind, mat->csr_col_ind,
                          mat->nnz * sizeof(int), cudaMemcpyHostToDevice));

    float *d_mat_csc_val;
    int *d_mat_csc_col_ptr, *d_mat_csc_row_ind;
    check_cuda(cudaMalloc(&d_mat_csc_val,mat->nnz * sizeof(float)));
    check_cuda(cudaMalloc(&d_mat_csc_col_ptr,(mat->columns + 1) * sizeof(int)));
    check_cuda(cudaMalloc(&d_mat_csc_row_ind,mat->nnz * sizeof(int)));

    size_t buffer_size;
    check_cusparse(cusparseCsr2cscEx2_bufferSize(cuda_helper->cusparse_handle,
                                                 mat->rows, mat->columns, mat->nnz,
                                                 d_mat_csr_val, d_mat_csr_row_ptr, d_mat_csr_col_ind,
                                                 d_mat_csc_val, d_mat_csc_col_ptr, d_mat_csc_row_ind,
                                                 CUDA_R_32F,
                                                 CUSPARSE_ACTION_SYMBOLIC, // could try CUSPARSE_ACTION_NUMERIC
                                                 CUSPARSE_INDEX_BASE_ZERO,
                                                 CUSPARSE_CSR2CSC_ALG1, // could try CUSPARSE_CSR2CSC_ALG2
                                                 &buffer_size));

    void *d_buffer;
    check_cuda(cudaMalloc(&d_buffer, buffer_size));

    check_cusparse(cusparseCsr2cscEx2(cuda_helper->cusparse_handle,
                                      mat->rows, mat->columns, mat->nnz,
                                      d_mat_csr_val, d_mat_csr_row_ptr, d_mat_csr_col_ind,
                                      d_mat_csc_val, d_mat_csc_col_ptr, d_mat_csc_row_ind,
                                      CUDA_R_32F,
                                      CUSPARSE_ACTION_SYMBOLIC, // could try CUSPARSE_ACTION_NUMERIC
                                      CUSPARSE_INDEX_BASE_ZERO,
                                      CUSPARSE_CSR2CSC_ALG1, // could try CUSPARSE_CSR2CSC_ALG2
                                      d_buffer));

    long tmp = mat->rows;
    mat->rows = mat->columns;
    mat->columns = tmp;
    delete mat->csr_row_ptr;
    delete mat->csr_col_ind;
    mat->csr_row_ptr = new int[mat->rows + 1];
    mat->csr_col_ind = new int[mat->nnz];

    check_cuda(cudaMemcpy(mat->csr_row_ptr, d_mat_csc_col_ptr,
                          (mat->rows + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    check_cuda(cudaMemcpy(mat->csr_col_ind, d_mat_csc_row_ind,
                          mat->nnz * sizeof(int), cudaMemcpyDeviceToHost));

    // free GPU memory
    check_cuda(cudaFree(d_buffer));
    check_cuda(cudaFree(d_mat_csc_row_ind));
    check_cuda(cudaFree(d_mat_csc_col_ptr));
    check_cuda(cudaFree(d_mat_csc_val));
    check_cuda(cudaFree(d_mat_csr_col_ind));
    check_cuda(cudaFree(d_mat_csr_row_ptr));
    check_cuda(cudaFree(d_mat_csr_val));
}

long count_nans(matrix<float> *x) {
    long num_nans = 0;

    for (int i = 0; i < x->rows * x->columns; ++i) {
        if (isnan(x->values[i])) {
            num_nans = num_nans + 1;
        }
    }

    return num_nans;
}

bool check_nans(matrix<float> *x, std::string name) {
    long num_nans = count_nans(x);
    if (num_nans > 0) {
        std::cout << name << " has " << num_nans << " NaNs" << std::endl;
        return true;
    } else {
        return false;
    }
}

matrix<float> gen_matrix(long num_rows, long num_columns, bool random) {
    long max = 5;

    matrix<float> mat = new_float_matrix(num_rows, num_columns, true);

    for (long i = 0; i < mat.rows * mat.columns; ++i) {
        if (random) {
            mat.values[i] = rand();
        } else {
            mat.values[i] = (float) ((i % max) + 1);
        }
    }

    return mat;
}

matrix<float> gen_rand_matrix(long num_rows, long num_columns) {
    return gen_matrix(num_rows, num_columns, true);
}

matrix<float> gen_non_rand_matrix(long num_rows, long num_columns) {
    return gen_matrix(num_rows, num_columns, false);
}
