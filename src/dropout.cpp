// Copyright 2020 Marcel Wagenländer

#include "dropout.hpp"
#include "cuda_helper.hpp"

#include <cuda_runtime.h>
#include <cudnn.h>


Dropout::Dropout(CudaHelper *helper) {
    cuda_helper_ = helper;
}


matrix<float> Dropout::forward(matrix<float> X) {
    float probability = 0.2f;
    size_t state_size;
    check_cudnn(cudnnDropoutGetStatesSize(cuda_helper_->cudnn_handle, &state_size));
    void *states;
    check_cuda(cudaMalloc(&states, state_size));
    unsigned long long seed = rand();
    cudnnDropoutDescriptor_t dropout_descr;
    check_cudnn(cudnnCreateDropoutDescriptor(&dropout_descr));
    check_cudnn(cudnnSetDropoutDescriptor(dropout_descr,
                                          cuda_helper_->cudnn_handle, probability,
                                          states, state_size, seed));

    matrix<float> Y;
    Y.rows = X.rows;
    Y.columns = X.columns;

    cudnnTensorDescriptor_t x_descr;
    check_cudnn(cudnnCreateTensorDescriptor(&x_descr));
    check_cudnn(cudnnSetTensor4dDescriptor(x_descr,
            CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
            1, 1, X.rows, X.columns));
    cudnnTensorDescriptor_t y_descr;
    check_cudnn(cudnnCreateTensorDescriptor(&y_descr));
    check_cudnn(cudnnSetTensor4dDescriptor(y_descr,
            CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
            1, 1, Y.rows, Y.columns));

    void *d_X, *d_Y;
    check_cuda(cudaMalloc(&d_X, X.rows * X.columns * sizeof(float)));
    check_cuda(cudaMemcpy(d_X, X.values, X.rows * X.columns * sizeof(float),
                          cudaMemcpyHostToDevice));
    check_cuda(cudaMalloc(&d_Y, Y.rows * Y.columns * sizeof(float)));


    Y.values = (float *) malloc(Y.rows * Y.columns * sizeof(float));
    check_cuda(cudaMemcpy(Y.values, d_Y, Y.rows * Y.columns * sizeof(float),
                          cudaMemcpyDeviceToHost));

    void *reserve_space;
    size_t reserve_space_size;
    check_cudnn(cudnnDropoutGetReserveSpaceSize(x_descr, &reserve_space_size));
    check_cuda(cudaMalloc(&reserve_space, reserve_space_size));
    check_cudnn(cudnnDropoutForward(cuda_helper_->cudnn_handle,
                                    dropout_descr, x_descr, d_X,
                                    y_descr, d_Y,
                                    reserve_space, reserve_space_size));

    check_cuda(cudaFree(states));
    check_cuda(cudaFree(reserve_space));
    check_cuda(cudaFree(d_X));
    check_cuda(cudaFree(d_Y));

    return Y;
}
