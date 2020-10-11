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
    alpha_ = 1.0;
    beta_ = 0.0;
}

matrix<float> Relu::forward(matrix<float> X) {
    float *d_X;
    check_cuda(cudaMalloc(&d_X, X.rows * X.columns * sizeof(float)));
    check_cuda(cudaMemcpy(d_X, X.values,
                          X.rows * X.columns * sizeof(float),
                          cudaMemcpyHostToDevice));
    cudnnTensorDescriptor_t x_desc;
    check_cudnn(cudnnCreateTensorDescriptor(&x_desc));
    check_cudnn(cudnnSetTensor4dDescriptor(x_desc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           1, 1, X.rows, X.columns));

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
    check_cudnn(cudnnSetTensor4dDescriptor(result_desc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           1, 1, result.rows, result.columns));

    cudnnActivationDescriptor_t relu_desc;
    check_cudnn(cudnnCreateActivationDescriptor(&relu_desc));
    double coef = std::numeric_limits<double>::max();
    check_cudnn(cudnnSetActivationDescriptor(relu_desc,
                                             CUDNN_ACTIVATION_RELU,
                                             CUDNN_PROPAGATE_NAN,
                                             coef));

    check_cudnn(cudnnActivationForward(cuda_helper_->cudnn_handle,
                                       relu_desc,
                                       &alpha_, x_desc, d_X,
                                       &beta_, result_desc, d_result));

    check_cuda(cudaMemcpy(result.values, d_result,
                          result.rows * result.columns * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // free GPU memory
    check_cuda(cudaFree(d_X));
    check_cuda(cudaFree(d_result));

    return result;
}

LogSoftmax::LogSoftmax(CudaHelper *helper) {
    cuda_helper_ = helper;
    alpha_ = 1.0;
    beta_ = 0.0;
}

matrix<float> LogSoftmax::forward(matrix<float> X) {
    float *d_X;
    check_cuda(cudaMalloc(&d_X, X.rows * X.columns * sizeof(float)));
    check_cuda(cudaMemcpy(d_X, X.values,
                          X.rows * X.columns * sizeof(float),
                          cudaMemcpyHostToDevice));
    cudnnTensorDescriptor_t x_desc;
    check_cudnn(cudnnCreateTensorDescriptor(&x_desc));
    check_cudnn(cudnnSetTensor4dDescriptor(x_desc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           X.rows, 1, 1, X.columns));

    matrix<float> Y;
    Y.rows = X.rows;
    Y.columns = X.columns;
    Y.values = (float *) malloc(Y.rows * Y.columns * sizeof(float));
    for (int i = 0; i < Y.rows * Y.columns; ++i) {
        Y.values[i] = 0.0;
    }
    float *d_y;
    check_cuda(cudaMalloc(&d_y, Y.rows * Y.columns * sizeof(float)));
    check_cuda(cudaMemcpy(d_y, Y.values,
                          Y.rows * Y.columns * sizeof(float),
                          cudaMemcpyHostToDevice));
    cudnnTensorDescriptor_t y_desc;
    check_cudnn(cudnnCreateTensorDescriptor(&y_desc));
    check_cudnn(cudnnSetTensor4dDescriptor(y_desc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           Y.rows, 1, 1, Y.columns));

    check_cudnn(cudnnSoftmaxForward(cuda_helper_->cudnn_handle,
                                    CUDNN_SOFTMAX_LOG,
                                    CUDNN_SOFTMAX_MODE_INSTANCE,
                                    &alpha_, x_desc, d_X,
                                    &beta_, y_desc, d_y));

    check_cuda(cudaMemcpy(Y.values, d_y,
                          Y.rows * Y.columns * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // free GPU memory
    check_cuda(cudaFree(d_X));
    check_cuda(cudaFree(d_y));

    Y_ = Y;

    return Y;
}

matrix<float> LogSoftmax::backward(matrix<float> in_gradients) {
    cudnnTensorDescriptor_t y_desc;
    float *d_y;
    check_cuda(cudaMalloc(&d_y, Y_.rows * Y_.columns * sizeof(float)));
    check_cuda(cudaMemcpy(d_y, Y_.values,
                          Y_.rows * Y_.columns * sizeof(float),
                          cudaMemcpyHostToDevice));
    check_cudnn(cudnnCreateTensorDescriptor(&y_desc));
    check_cudnn(cudnnSetTensor4dDescriptor(y_desc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           Y_.rows, 1, 1, Y_.columns));

    cudnnTensorDescriptor_t dy_desc;
    float *d_dy;
    check_cuda(cudaMalloc(&d_dy, Y_.rows * Y_.columns * sizeof(float)));
    check_cuda(cudaMemcpy(d_dy, in_gradients.values,
                          in_gradients.rows * in_gradients.columns * sizeof(float),
                          cudaMemcpyHostToDevice));
    check_cudnn(cudnnCreateTensorDescriptor(&dy_desc));
    check_cudnn(cudnnSetTensor4dDescriptor(dy_desc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           in_gradients.rows, 1, 1, in_gradients.columns));

    cudnnTensorDescriptor_t dx_desc;
    float *d_dx;
    check_cuda(cudaMalloc(&d_dx, Y_.rows * Y_.columns * sizeof(float)));
    check_cudnn(cudnnCreateTensorDescriptor(&dx_desc));
    check_cudnn(cudnnSetTensor4dDescriptor(dx_desc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           Y_.rows, 1, 1, Y_.columns));

    check_cudnn(cudnnSoftmaxBackward(cuda_helper_->cudnn_handle,
                                     CUDNN_SOFTMAX_LOG,
                                     CUDNN_SOFTMAX_MODE_INSTANCE,
                                     &alpha_, y_desc, d_y,
                                     dy_desc, d_dy,
                                     &beta_, dx_desc, d_dx));

    matrix<float> gradients;
    gradients.rows = Y_.rows;
    gradients.columns = Y_.columns;
    gradients.values = reinterpret_cast<float *>(
            malloc(gradients.rows * gradients.columns * sizeof(float)));
    check_cuda(cudaMemcpy(gradients.values, d_dx,
                          gradients.rows * gradients.columns * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // free GPU memory
    check_cuda(cudaFree(d_y));
    check_cuda(cudaFree(d_dy));
    check_cuda(cudaFree(d_dx));

    return gradients;
}
