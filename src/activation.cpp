// Copyright 2020 Marcel Wagenländer

#include "activation.hpp"
#include "cuda_helper.hpp"

#include <limits>
#include <cuda_runtime.h>
#include <cudnn.h>


matrix<float> cudnn_forward(matrix<float> X, char mode) {
    char relu_mode = 'r';
    char softmax_mode = 's';
    cudaError_t cuda_error;
    cudnnStatus_t cudnn_status;
    cudnnHandle_t cudnn_handle;
    cudnn_status = cudnnCreate(&cudnn_handle);
    check_cudnn(cudnn_status);

    float *d_X;
    cuda_error = cudaMalloc(&d_X, X.rows * X.columns * sizeof(float));
    check_cuda(cuda_error);
    cuda_error = cudaMemcpy(d_X, X.values,
            X.rows * X.columns * sizeof(float),
            cudaMemcpyHostToDevice);
    check_cuda(cuda_error);
    cudnnTensorDescriptor_t x_desc;
    cudnn_status = cudnnCreateTensorDescriptor(&x_desc);
    check_cudnn(cudnn_status);
    if (mode == relu_mode) {
        cudnn_status = cudnnSetTensor4dDescriptor(x_desc,
                CUDNN_TENSOR_NCHW,
                CUDNN_DATA_FLOAT,
                1, 1, X.rows, X.columns);
    } else if (mode == softmax_mode) {
        cudnn_status = cudnnSetTensor4dDescriptor(x_desc,
                CUDNN_TENSOR_NCHW,
                CUDNN_DATA_FLOAT,
                X.rows, 1, 1, X.columns);
    }
    check_cudnn(cudnn_status);
    
    matrix<float> result;
    result.rows = X.rows;
    result.columns = X.columns;
    result.values = (float *) malloc(result.rows * result.columns * sizeof(float));
    for (int i = 0; i < result.rows * result.columns; ++i) {
        result.values[i] = 0.0;
    }
    float *d_result;
    cuda_error = cudaMalloc(&d_result, result.rows * result.columns * sizeof(float));
    check_cuda(cuda_error);
    cuda_error = cudaMemcpy(d_result, result.values,
            result.rows * result.columns * sizeof(float),
            cudaMemcpyHostToDevice);
    check_cuda(cuda_error);
    cudnnTensorDescriptor_t result_desc;
    cudnn_status = cudnnCreateTensorDescriptor(&result_desc);
    check_cudnn(cudnn_status);
    if (mode == relu_mode) {
        cudnn_status = cudnnSetTensor4dDescriptor(result_desc,
                CUDNN_TENSOR_NCHW,
                CUDNN_DATA_FLOAT,
                1, 1, result.rows, result.columns);
    } else if (mode == softmax_mode) {
        cudnn_status = cudnnSetTensor4dDescriptor(result_desc,
                CUDNN_TENSOR_NCHW,
                CUDNN_DATA_FLOAT,
                result.rows, 1, 1, result.columns);
    }
    check_cudnn(cudnn_status);

    float alpha = 1.0;
    float beta = 0.0;
    if (mode == relu_mode) {
        cudnnActivationDescriptor_t relu_desc;
        cudnn_status = cudnnCreateActivationDescriptor(&relu_desc);
        check_cudnn(cudnn_status);
        double coef = std::numeric_limits<double>::max();
        cudnn_status = cudnnSetActivationDescriptor(relu_desc,
                CUDNN_ACTIVATION_RELU,
                CUDNN_PROPAGATE_NAN,
                coef);
        check_cudnn(cudnn_status);
    
        cudnn_status = cudnnActivationForward(cudnn_handle,
                relu_desc,
                &alpha, x_desc, d_X,
                &beta, result_desc, d_result);
        check_cudnn(cudnn_status);
    } else if (mode == softmax_mode) {
        cudnn_status = cudnnSoftmaxForward(cudnn_handle,
                                           CUDNN_SOFTMAX_LOG,
                CUDNN_SOFTMAX_MODE_INSTANCE,
                &alpha, x_desc, d_X,
                &beta, result_desc, d_result);
        check_cudnn(cudnn_status);
    }

    cuda_error = cudaMemcpy(result.values, d_result,
            result.rows * result.columns * sizeof(float),
            cudaMemcpyDeviceToHost);
    check_cuda(cuda_error);

    // free GPU memory
    cuda_error = cudaFree(d_X);
    check_cuda(cuda_error);
    cuda_error = cudaFree(d_result);
    check_cuda(cuda_error);

    // clean cudnn
    cudnn_status = cudnnDestroy(cudnn_handle);
    check_cudnn(cudnn_status);

    return result;
}

matrix<float> relu(matrix<float> X) {
    return cudnn_forward(X, 'r');
}

matrix<float> softmax(matrix<float> X) {
    return cudnn_forward(X, 's');
}
