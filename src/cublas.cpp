#include <iostream>
#include <stdlib.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>


void init_matrix(float *a, const int kN) {
    for (int i = 0; i < kN; i = i + 1) {
        a[i] = rand() / static_cast<float>(RAND_MAX);
    }
}

void print_matrix(float *a, int rows, int cols) {
    for (int i = 0; i < rows; i = i + 1) {
        for (int j = 0; j < cols; j = j + 1) {
            std::cout << a[i * cols + j] << "\n";
        }
        std::cout << "---\n";
    }
}


int main(int argc, char **argv) {
    cublasStatus_t status;
    cublasHandle_t handle;
    cudaError_t error;
    const int kM = 128;
    const int kN = 256;
    const int kK = 512;
    float alpha, beta = 1.0f;

    float *cpu_A = reinterpret_cast<float *>(malloc(kM * kK * sizeof(float)));
    float *cpu_B = reinterpret_cast<float *>(malloc(kK * kN * sizeof(float)));
    float *cpu_C = reinterpret_cast<float *>(malloc(kM * kN * sizeof(float)));
    init_matrix(cpu_A, kM * kK);
    init_matrix(cpu_B, kK * kN);
    init_matrix(cpu_C, kM * kN);
    
    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "cublasCreate failed\n";
    }

    float *gpu_A, *gpu_B, *gpu_C;
    error = cudaMalloc(reinterpret_cast<void **>(&gpu_A), kM * kK * sizeof(float));
    if (error != cudaSuccess) {
        std::cout << "cudaMalloc gpu_A failed\n";
    }
    error = cudaMalloc(reinterpret_cast<void **>(&gpu_B), kK * kN * sizeof(float));
    if (error != cudaSuccess) {
        std::cout << "cudaMalloc gpu_B failed\n";
    }
    error = cudaMalloc(reinterpret_cast<void **>(&gpu_C), kM * kN * sizeof(float));
    if (error != cudaSuccess) {
        std::cout << "cudaMalloc gpu_C failed\n";
    }
    status = cublasSetMatrix(kM, kK, sizeof(float), cpu_A, kM, gpu_A, kM);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "cublasSetMatrix gpu_A failed\n";
    }
    status = cublasSetMatrix(kK, kN, sizeof(float), cpu_B, kK, gpu_B, kK);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "cublasSetMatrix gpu_B failed\n";
    }
    status = cublasSetMatrix(kM, kN, sizeof(float), cpu_C, kM, gpu_C, kM);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "cublasSetMatrix gpu_C failed\n";
    }

    // use tensor cores
    status = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "cublasSetMathMode failed\n";
    }

    // cublasStatus_t cublasSgemm(cublasHandle_t handle,
    //                            cublasOperation_t transa, cublasOperation_t transb,
    //                            int m, int n, int k,
    //                            const float *alpha,
    //                            const float *A, int lda,
    //                            const float *B, int ldb,
    //                            const float *beta,
    //                            float *C, int ldc)

    cublasOperation_t operation_t = CUBLAS_OP_N; 
    status = cublasSgemm(handle, operation_t, operation_t,
                         kM, kN, kK,
                         &alpha,
                         gpu_A, kM,
                         gpu_B, kK,
                         &beta,
                         gpu_C, kM);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "cublasSgemm failed\n";
    }

    status = cublasGetMatrix(kM, kN, sizeof(float), gpu_C, kM, cpu_C, kM);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "cublasGetMatrix failed\n";
    }

    print_matrix(cpu_C, kM, kN);

    free(cpu_A);
    free(cpu_B);
    free(cpu_C);

    status = cublasDestroy(handle);
    
    error = cudaFree(gpu_A);
    error = cudaFree(gpu_B);
    error = cudaFree(gpu_C);

}

