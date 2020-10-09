// Copyright 2020 Marcel Wagenländer

#include "activation.hpp"
#include "cuda_helper.hpp"

#include <limits>
#include <cuda_runtime.h>
#include <cudnn.h>


matrix<float> cudnn_forward(matrix<float> X, char mode, CudaHelper *cuda_helper) {
    char relu_mode = 'r';
    char softmax_mode = 's';

    float *d_X;
    check_cuda(cudaMalloc(&d_X, X.rows * X.columns * sizeof(float)));
    check_cuda(cudaMemcpy(d_X, X.values,
                          X.rows * X.columns * sizeof(float),
                          cudaMemcpyHostToDevice));
    cudnnTensorDescriptor_t x_desc;
    check_cudnn(cudnnCreateTensorDescriptor(&x_desc));
    if (mode == relu_mode) {
        check_cudnn(cudnnSetTensor4dDescriptor(x_desc,
                                               CUDNN_TENSOR_NCHW,
                                               CUDNN_DATA_FLOAT,
                                               1, 1, X.rows, X.columns));
    } else if (mode == softmax_mode) {
        check_cudnn(cudnnSetTensor4dDescriptor(x_desc,
                                               CUDNN_TENSOR_NCHW,
                                               CUDNN_DATA_FLOAT,
                                               X.rows, 1, 1, X.columns));
    }

    matrix<float> result;
    result.rows = X.rows;
    result.columns = X.columns;
    result.values = (float *) malloc(result.rows * result.columns * sizeof(float));
    for (int i = 0; i < result.rows * result.columns; ++i) {
        result.values[i] = 0.0;
    }
    float *d_result;
    check_cuda(cudaMalloc(&d_result, result.rows * result.columns * sizeof(float)));
    check_cuda(cudaMemcpy(d_result, result.values,
                          result.rows * result.columns * sizeof(float),
                          cudaMemcpyHostToDevice));
    cudnnTensorDescriptor_t result_desc;
    check_cudnn(cudnnCreateTensorDescriptor(&result_desc));
    if (mode == relu_mode) {
        check_cudnn(cudnnSetTensor4dDescriptor(result_desc,
                                               CUDNN_TENSOR_NCHW,
                                               CUDNN_DATA_FLOAT,
                                               1, 1, result.rows, result.columns));
    } else if (mode == softmax_mode) {
        check_cudnn(cudnnSetTensor4dDescriptor(result_desc,
                                               CUDNN_TENSOR_NCHW,
                                               CUDNN_DATA_FLOAT,
                                               result.rows, 1, 1, result.columns));
    }

    float alpha = 1.0;
    float beta = 0.0;
    if (mode == relu_mode) {
        cudnnActivationDescriptor_t relu_desc;
        check_cudnn(cudnnCreateActivationDescriptor(&relu_desc));
        double coef = std::numeric_limits<double>::max();
        check_cudnn(cudnnSetActivationDescriptor(relu_desc,
                                                 CUDNN_ACTIVATION_RELU,
                                                 CUDNN_PROPAGATE_NAN,
                                                 coef));

        check_cudnn(cudnnActivationForward(cuda_helper->cudnn_handle,
                                           relu_desc,
                                           &alpha, x_desc, d_X,
                                           &beta, result_desc, d_result));
    } else if (mode == softmax_mode) {
        check_cudnn(cudnnSoftmaxForward(cuda_helper->cudnn_handle,
                                        CUDNN_SOFTMAX_LOG,
                                        CUDNN_SOFTMAX_MODE_INSTANCE,
                                        &alpha, x_desc, d_X,
                                        &beta, result_desc, d_result));
    }

    check_cuda(cudaMemcpy(result.values, d_result,
                          result.rows * result.columns * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // free GPU memory
    check_cuda(cudaFree(d_X));
    check_cuda(cudaFree(d_result));

    return result;
}

Relu::Relu(CudaHelper *helper) {
    cuda_helper_ = helper;
}

matrix<float> Relu::forward(matrix<float> X) {
    return cudnn_forward(X, 'r', cuda_helper_);
}

LogSoftmax::LogSoftmax(CudaHelper *helper) {
    cuda_helper_ = helper;
}

matrix<float> LogSoftmax::forward(matrix<float> X) {
    return cudnn_forward(X, 's', cuda_helper_);
}
