// 2020 Marcel Wagenländer

#ifndef ALZHEIMER_SPARSE_COMPUTATION_H
#define ALZHEIMER_SPARSE_COMPUTATION_H

#include "cuda_helper.hpp"
#include "tensors.hpp"


void sp_mat_mat_multi(CudaHelper *cuda_helper, SparseMatrix<float> *sp_mat, Matrix<float> *mat, Matrix<float> *result);

void sp_mat_sum_rows(CudaHelper *cuda_helper, SparseMatrix<float> *sp_mat, Matrix<float> *sum);

#endif//ALZHEIMER_SPARSE_COMPUTATION_H
