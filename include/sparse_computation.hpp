// 2020 Marcel Wagenländer

#ifndef ALZHEIMER_SPARSE_COMPUTATION_H
#define ALZHEIMER_SPARSE_COMPUTATION_H

#include "cuda_helper.hpp"
#include "tensors.hpp"


void malloc_memcpy_sp_mat(SparseMatrixCuda<float> *d_sp_mat, SparseMatrix<float> *sp_mat);

void sp_mat_mat_multi_cuda(CudaHelper *cuda_helper, SparseMatrixCuda<float> *d_sp_mat, float *d_mat, float *d_result, long mat_columns, bool add_to_result);

void sp_mat_mat_multi(CudaHelper *cuda_helper, SparseMatrix<float> *sp_mat, Matrix<float> *mat, Matrix<float> *result, bool add_to_result);

void sp_mat_sum_rows(CudaHelper *cuda_helper, SparseMatrix<float> *sp_mat, Matrix<float> *sum);

void sp_mat_sum_rows(SparseMatrix<float> *sp_mat, Matrix<float> *sum);

#endif//ALZHEIMER_SPARSE_COMPUTATION_H
