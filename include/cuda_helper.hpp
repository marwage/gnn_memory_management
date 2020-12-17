// Copyright 2020 Marcel Wagenländer

#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

#include "cusparse.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudnn.h>

class CudaHelper {
public:
    cublasHandle_t cublas_handle;
    cudnnHandle_t cudnn_handle;
    cusparseHandle_t cusparse_handle;

    cudaStream_t stream_in_;
    cudaStream_t stream_out_;
    cudaStream_t stream_compute_;

    CudaHelper();
    ~CudaHelper();
};


void check_cuda(cudaError_t status);

void check_cusparse(cusparseStatus_t status);

void check_cudnn(cudnnStatus_t status);

void check_cublas(cublasStatus_t status);

#endif
