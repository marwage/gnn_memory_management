#include "cuda_helper.hpp"

#include <string>


CudaHelper::CudaHelper() {
    check_cublas(cublasCreate(&cublas_handle));
    check_cudnn(cudnnCreate(&cudnn_handle));
    check_cusparse(cusparseCreate(&cusparse_handle));

    check_cuda(cudaStreamCreate(&stream_in_));
    check_cuda(cudaStreamCreate(&stream_out_));
    check_cuda(cudaStreamCreate(&stream_compute_));

    check_cublas(cublasSetStream(cublas_handle, stream_compute_));
    check_cudnn(cudnnSetStream(cudnn_handle, stream_compute_));
    check_cusparse(cusparseSetStream(cusparse_handle, stream_compute_));
}

CudaHelper::~CudaHelper() {
    check_cublas(cublasDestroy(cublas_handle));
    check_cudnn(cudnnDestroy(cudnn_handle));
    check_cusparse(cusparseDestroy(cusparse_handle));

    check_cuda(cudaStreamDestroy(stream_in_));
    check_cuda(cudaStreamDestroy(stream_out_));
    check_cuda(cudaStreamDestroy(stream_compute_));
}

void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw(std::string) "CUDA API failed with error: " + (std::string) cudaGetErrorString(status);
    }
}

void check_cusparse(cusparseStatus_t status) {
    if (status != CUSPARSE_STATUS_SUCCESS) {
        throw(std::string) "CUSPARSE API failed with error: " + (std::string) cusparseGetErrorString(status);
    }
}

void check_cudnn(cudnnStatus_t status) {
    if (status != CUDNN_STATUS_SUCCESS) {
        throw(std::string) "CUDNN API failed with error: " + (std::string) cudnnGetErrorString(status);
    }
}

const char *cublasGetErrorString(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "unknown error";
}

void check_cublas(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw(std::string) "CUBLAS API failed with error: " + (std::string) cublasGetErrorString(status);
    }
}
