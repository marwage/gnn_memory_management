// Copyright 2020 Marcel Wagenländer

#include "tensors.hpp"

#include "cnpy.h"
#include "mmio_wrapper.hpp"

#include <cstring>
#include <iostream>
#include <thread>


template<typename T>
void print_matrix(matrix<T> mat) {
    int N;
    if (mat.rows < 10) {
        N = mat.rows;
    } else {
        N = 10;
    }
    int M;
    if (mat.columns < 10) {
        M = mat.columns;
    } else {
        M = 10;
    }

    std::cout << "-----" << std::endl;
    // for (int i = 0; i < rows; i = i + 1) {
    for (int i = 0; i < N; i = i + 1) {
        // for (int j = 0; j < cols; j = j + 1) {
        for (int j = 0; j < M; j = j + 1) {
            std::cout << mat.values[j * mat.rows + i] << ",";
        }
        std::cout << std::endl;
    }
}

template void print_matrix<float>(matrix<float> mat);
template void print_matrix<int>(matrix<int> mat);


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
void transpose(T *a, T *a_T, long rows, long cols,
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
            std::thread thread_one(transpose<T>, a, a_T, rows, cols, rows_lower, rows_upper, columns_lower, column_mid);
            std::thread thread_two(transpose<T>, a, a_T, rows, cols, rows_lower, rows_upper, column_mid, columns_upper);
            thread_one.join();
            thread_two.join();
        }
    } else {
        if (columns_upper - columns_lower < boundary) {
            int row_mid = (rows_upper - rows_lower) / 2 + rows_lower;
            std::thread thread_one(transpose<T>, a, a_T, rows, cols, rows_lower, row_mid, columns_lower, columns_upper);
            std::thread thread_two(transpose<T>, a, a_T, rows, cols, row_mid, rows_upper, columns_lower, columns_upper);
            thread_one.join();
            thread_two.join();
        } else {
            int row_mid = (rows_upper - rows_lower) / 2 + rows_lower;
            int column_mid = (columns_upper - columns_lower) / 2 + columns_lower;
            std::thread thread_one(transpose<T>, a, a_T, rows, cols, rows_lower, row_mid, columns_lower, column_mid);
            std::thread thread_two(transpose<T>, a, a_T, rows, cols, rows_lower, row_mid, column_mid, columns_upper);
            std::thread thread_three(transpose<T>, a, a_T, rows, cols, row_mid, rows_upper, columns_lower, column_mid);
            std::thread thread_four(transpose<T>, a, a_T, rows, cols, row_mid, rows_upper, column_mid, columns_upper);
        }
    }
}

template<typename T>
T *transpose(T *a, long rows, long cols) {

    T *a_T = (T *) malloc(rows * cols * sizeof(T));

    transpose(a, a_T, rows, cols, 0, rows, 0, cols);

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
    mat.values = reinterpret_cast<T *>(
            malloc(mat.rows * mat.columns * sizeof(T)));
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
void save_npy_matrix(matrix<T> mat, std::string path) {
    to_row_major_inplace(&mat);
    std::vector<size_t> shape = {(size_t) mat.rows, (size_t) mat.columns};
    cnpy::npy_save<T>(path, mat.values, shape);
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
void to_column_major_inplace(matrix<T> *mat, bool free_mem) {
    if (mat->row_major) {
        T *new_values = transpose<T>(mat->values, mat->rows, mat->columns);
//        if (free_mem) free(mat->values);
        mat->values = new_values;
        mat->row_major = false;
    }
}

template void to_column_major_inplace<float>(matrix<float> *mat, bool free_mem);
template void to_column_major_inplace<int>(matrix<int> *mat, bool free_mem);

template<typename T>
void to_column_major_inplace(matrix<T> *mat) {
    to_column_major_inplace(mat, true);
}

template void to_column_major_inplace<float>(matrix<float> *mat);
template void to_column_major_inplace<int>(matrix<int> *mat);

template<typename T>
matrix<T> to_column_major(matrix<T> *mat) {
    if (mat->row_major) {
        matrix<T> mat_transposed = *mat;
        to_column_major_inplace(&mat_transposed, false);
        return mat_transposed;
    } else {
        return *mat;
    }
}

template matrix<float> to_column_major<float>(matrix<float> *mat);
template matrix<int> to_column_major<int>(matrix<int> *mat);

template<typename T>
void to_row_major_inplace(matrix<T> *mat, bool free_mem) {
    if (!mat->row_major) {
        T *new_values = transpose<T>(mat->values, mat->columns, mat->rows);
//        if (free_mem) free(mat->values);
        mat->values = new_values;
        mat->row_major = true;
    }
}

template void to_row_major_inplace<float>(matrix<float> *mat, bool free_mem);
template void to_row_major_inplace<int>(matrix<int> *mat, bool free_mem);

template<typename T>
void to_row_major_inplace(matrix<T> *mat) {
    to_row_major_inplace(mat, true);
}

template void to_row_major_inplace<float>(matrix<float> *mat);
template void to_row_major_inplace<int>(matrix<int> *mat);

template<typename T>
matrix<T> to_row_major(matrix<T> *mat) {
    if (mat->row_major) {
        return *mat;
    } else {
        matrix<T> mat_transposed = *mat;
        to_row_major_inplace(&mat_transposed, false);
        return mat_transposed;
    }
}

template matrix<float> to_row_major<float>(matrix<float> *mat);
template matrix<int> to_row_major<int>(matrix<int> *mat);


matrix<float> add_matrices(CudaHelper *cuda_helper, matrix<float> mat_a, matrix<float> mat_b) {
    float alpha = 1.0;

    float *d_a;
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_a),
                          mat_a.rows * mat_a.columns * sizeof(float)));
    check_cuda(cudaMemcpy(d_a, mat_a.values,
                          mat_a.rows * mat_a.columns * sizeof(float),
                          cudaMemcpyHostToDevice));

    float *d_b;
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_b),
                          mat_b.rows * mat_b.columns * sizeof(float)));
    check_cuda(cudaMemcpy(d_b, mat_b.values,
                          mat_b.rows * mat_b.columns * sizeof(float),
                          cudaMemcpyHostToDevice));

    check_cublas(cublasSaxpy(cuda_helper->cublas_handle,
                             mat_a.rows * mat_a.columns,
                             &alpha, d_a, 1,
                             d_b, 1));

    matrix<float> mat_c;
    mat_c.rows = mat_a.rows;
    mat_c.columns = mat_a.columns;
    mat_c.values = reinterpret_cast<float *>(malloc(mat_c.rows * mat_c.columns * sizeof(float)));

    check_cuda(cudaMemcpy(mat_c.values, d_b,
                          mat_c.rows * mat_c.columns * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // clean-up
    check_cuda(cudaFree(d_a));
    check_cuda(cudaFree(d_b));

    return mat_c;
}

sparse_matrix<float> get_rows(sparse_matrix<float> mat, int start_row, int end_row) {
    sparse_matrix<float> reduced_mat;
    reduced_mat.rows = end_row - start_row;
    reduced_mat.columns = mat.columns;

    int first_index = mat.csr_row_ptr[start_row];
    int last_index = mat.csr_row_ptr[end_row];
    reduced_mat.nnz = last_index - first_index;
    reduced_mat.csr_val = (float *) malloc(reduced_mat.nnz * sizeof(float));
    reduced_mat.csr_row_ptr = (int *) malloc((reduced_mat.rows + 1) * sizeof(int));
    reduced_mat.csr_col_ind = (int *) malloc(reduced_mat.nnz * sizeof(int));

    std::memcpy(reduced_mat.csr_val, &mat.csr_val[first_index], reduced_mat.nnz * sizeof(float));
    std::memcpy(reduced_mat.csr_row_ptr, &mat.csr_row_ptr[start_row], reduced_mat.rows * sizeof(int));
    std::memcpy(reduced_mat.csr_col_ind, &mat.csr_col_ind[first_index], reduced_mat.nnz * sizeof(int));
    for (int i = 0; i < reduced_mat.rows; ++i) {
        reduced_mat.csr_row_ptr[i] = reduced_mat.csr_row_ptr[i] - first_index;
    }

    return reduced_mat;
}

void print_sparse_matrix(sparse_matrix<float> mat) {
    std::cout << "Row pointers" << std::endl;
    for (int i = 0; i < mat.rows + 1; ++i) {
        std::cout << mat.csr_row_ptr[i] << ", ";
    }
    std::cout << std::endl;
    std::cout << "Column indices" << std::endl;
    for (int i = 0; i < mat.nnz; ++i) {
        std::cout << mat.csr_col_ind[i] << ", ";
    }
    std::cout << std::endl;
    std::cout << "Values" << std::endl;
    for (int i = 0; i < mat.nnz; ++i) {
        std::cout << mat.csr_val[i] << ", ";
    }
    std::cout << std::endl;
}
