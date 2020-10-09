#include "cuda_helper.hpp"

#include <string>


CudaHelper::CudaHelper() {
    cublas_status = cublasCreate(&cublas_handle);
    check_cublas(cublas_status);
    cudnn_status = cudnnCreate(&cudnn_handle);
    check_cudnn(cudnn_status);
    cusparse_status = cusparseCreate(&cusparse_handle);
    check_cusparse(cusparse_status);
}

void CudaHelper::destroy_handles() {
    cublas_status = cublasDestroy(cublas_handle);
    check_cublas(cublas_status);
    cudnn_status = cudnnDestroy(cudnn_handle);
    check_cudnn(cudnn_status);
    cusparse_status = cusparseDestroy(cusparse_handle);
    check_cusparse(cusparse_status);
}

void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw (std::string) "CUDA API failed with error: " + (std::string) cudaGetErrorString(status);
    }
}

void check_cusparse(cusparseStatus_t status) {
    if (status != CUSPARSE_STATUS_SUCCESS) {
        throw (std::string) "CUSPARSE API failed with error: " + (std::string) cusparseGetErrorString(status);
    }
}

void check_cudnn(cudnnStatus_t status) {
    if (status != CUDNN_STATUS_SUCCESS) {
        throw (std::string) "CUDNN API failed with error: " + (std::string) cudnnGetErrorString(status);
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
        throw (std::string) "CUBLAS API failed with error: " + (std::string) cublasGetErrorString(status);
    }
}
