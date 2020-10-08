// Copyright 2020 Marcel Wagenländer

#include "dropout.hpp"
#include "cuda_helper.hpp"

#include <cuda_runtime.h>
#include <cudnn.h>


matrix<float> dropout(matrix<float> X) {
    cudaError_t cuda_error;
    cudnnStatus_t cudnn_status;
    cudnnHandle_t cudnn_handle;
    cudnn_status = cudnnCreate(&cudnn_handle);

    float probability = 0.2f;
    size_t state_size;
    cudnn_status = cudnnDropoutGetStatesSize(cudnn_handle, &state_size);
    void *states;
    cuda_error = cudaMalloc(&states, state_size);
    check_cuda(cuda_error);
    unsigned long long seed = rand();
    cudnnDropoutDescriptor_t dropout_descr;
    cudnn_status = cudnnCreateDropoutDescriptor(&dropout_descr);
    check_cudnn(cudnn_status);
    cudnn_status = cudnnSetDropoutDescriptor(dropout_descr,
            cudnn_handle, probability,
            states, state_size, seed);
    check_cudnn(cudnn_status);

    matrix<float> Y;
    Y.rows = X.rows;
    Y.columns = X.columns;

    cudnnTensorDescriptor_t x_descr;
    cudnn_status = cudnnCreateTensorDescriptor(&x_descr);
    check_cudnn(cudnn_status);
    cudnn_status = cudnnSetTensor4dDescriptor(x_descr,
            CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
            1, 1, X.rows, X.columns);
    check_cudnn(cudnn_status);
    cudnnTensorDescriptor_t y_descr;
    cudnn_status = cudnnCreateTensorDescriptor(&y_descr);
    check_cudnn(cudnn_status);
    cudnn_status = cudnnSetTensor4dDescriptor(y_descr,
            CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
            1, 1, Y.rows, Y.columns);
    check_cudnn(cudnn_status);
    void *d_X, *d_Y;
    // cudaMemcpy
    cuda_error = cudaMalloc(&d_X, X.rows * X.columns * sizeof(float));
    check_cuda(cuda_error);
    cuda_error = cudaMemcpy(d_X, X.values, X.rows * X.columns * sizeof(float),
            cudaMemcpyHostToDevice);
    check_cuda(cuda_error);
    cuda_error = cudaMalloc(&d_Y, Y.rows * Y.columns * sizeof(float));
    check_cuda(cuda_error);
    void *reserve_space;
    size_t reserve_space_size;
    cudnn_status = cudnnDropoutGetReserveSpaceSize(x_descr, &reserve_space_size);
    check_cudnn(cudnn_status);
    cuda_error = cudaMalloc(&reserve_space, reserve_space_size);
    check_cuda(cuda_error);
    cudnn_status = cudnnDropoutForward(cudnn_handle,
            dropout_descr, x_descr, d_X,
            y_descr, d_Y,
            reserve_space, reserve_space_size);
    check_cudnn(cudnn_status);

    Y.values = (float *) malloc(Y.rows * Y.columns * sizeof(float));
    cuda_error = cudaMemcpy(Y.values, d_Y, Y.rows * Y.columns * sizeof(float),
            cudaMemcpyDeviceToHost);
    check_cuda(cuda_error);

    cudnn_status = cudnnDestroy(cudnn_handle);

    cuda_error = cudaFree(states);
    check_cuda(cuda_error);
    cuda_error = cudaFree(reserve_space);
    check_cuda(cuda_error);
    cuda_error = cudaFree(d_X);
    check_cuda(cuda_error);
    cuda_error = cudaFree(d_Y);
    check_cuda(cuda_error);

    return Y;
}
