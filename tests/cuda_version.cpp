// 2021 Copyright Marcel Wagenländer

#include "cusparse.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <catch2/catch.hpp>


int cuda_version() {
    cublasHandle_t cublas_handle;
    cudnnHandle_t cudnn_handle;
    cusparseHandle_t cusparse_handle;

    cublasCreate(&cublas_handle);
    cudnnCreate(&cudnn_handle);
    cusparseCreate(&cusparse_handle);

    int cusparse_version;
    int cublas_version;
    int cuda_runtime_version;
    cusparseGetVersion(cusparse_handle, &cusparse_version);
    cublasGetVersion(cublas_handle, &cublas_version);
    cudaRuntimeGetVersion(&cuda_runtime_version);
    size_t cudnn_version = cudnnGetVersion();

    std::cout << "Cuda runtime version is " << cuda_runtime_version << std::endl;
    std::cout << "cuSparse version is " << cusparse_version << std::endl;
    std::cout << "cuBlas version is " << cublas_version << std::endl;
    std::cout << "cuDnn version is " << cudnn_version << std::endl;

    return 1;
}

TEST_CASE("Cuda version", "[cuda]") {
    CHECK(cuda_version());
}
