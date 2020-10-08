// Copyright 2020 Marcel Wagenländer

#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

#include <cuda_runtime.h>
#include "cusparse.h"
#include <cudnn.h>
#include <cublas_v2.h>


void check_cuda(cudaError_t status);

void check_cusparse(cusparseStatus_t status);

void check_cudnn(cudnnStatus_t status);

void check_cublas(cublasStatus_t status);

#endif
