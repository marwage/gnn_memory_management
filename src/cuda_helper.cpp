#include "cuda_helper.hpp"

#include <stdio.h>


void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) {
        printf("CUDA API failed with error: %s (%d)\n",
                cudaGetErrorString(status), status);
    }
}

void check_cusparse(cusparseStatus_t status) {
    if (status != CUSPARSE_STATUS_SUCCESS) {
        printf("CUSPARSE API failed with error: %s (%d)\n",
                cusparseGetErrorString(status), status);
    }
}

void check_cudnn(cudnnStatus_t status) {
    if (status != CUDNN_STATUS_SUCCESS) {
        printf("CUDNN API failed with error: %s (%d)\n",
                cudnnGetErrorString(status), status);
    }
}

const char* cublasGetErrorString(cublasStatus_t status) {
    switch(status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    }
    return "unknown error";
}

void check_cublas(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS API failed with error: %s (%d)\n",
                cublasGetErrorString(status), status);
    }
}
