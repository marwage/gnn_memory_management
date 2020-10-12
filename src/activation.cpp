// Copyright 2020 Marcel Wagenländer

#include "activation.hpp"
#include "cuda_helper.hpp"

#include <limits>
#include <cuda_runtime.h>
#include <cudnn.h>


Relu::Relu(CudaHelper *helper) {
    cuda_helper_ = helper;
    alpha_ = 1.0;
    beta_ = 0.0;
}

matrix<float> Relu::forward(matrix<float> X) {
    x_ = X;
    float *d_X;
    check_cuda(cudaMalloc(&d_X, X.rows * X.columns * sizeof(float)));
    check_cuda(cudaMemcpy(d_X, X.values,
                          X.rows * X.columns * sizeof(float),
                          cudaMemcpyHostToDevice));
    check_cudnn(cudnnCreateTensorDescriptor(&x_desc_));
    check_cudnn(cudnnSetTensor4dDescriptor(x_desc_,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           1, 1, X.rows, X.columns));

    y_.rows = X.rows;
    y_.columns = X.columns;
    y_.values = (float *) malloc(y_.rows * y_.columns * sizeof(float));
    for (int i = 0; i < y_.rows * y_.columns; ++i) {
        y_.values[i] = 0.0;
    }
    float *d_y;
    check_cuda(cudaMalloc(&d_y, y_.rows * y_.columns * sizeof(float)));
    check_cuda(cudaMemcpy(d_y, y_.values,
                          y_.rows * y_.columns * sizeof(float),
                          cudaMemcpyHostToDevice));
    check_cudnn(cudnnCreateTensorDescriptor(&y_desc_));
    check_cudnn(cudnnSetTensor4dDescriptor(y_desc_,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           1, 1, y_.rows, y_.columns));

    check_cudnn(cudnnCreateActivationDescriptor(&relu_desc_));
    double coef = std::numeric_limits<double>::max();
    check_cudnn(cudnnSetActivationDescriptor(relu_desc_,
                                             CUDNN_ACTIVATION_RELU,
                                             CUDNN_PROPAGATE_NAN,
                                             coef));

    check_cudnn(cudnnActivationForward(cuda_helper_->cudnn_handle,
                                       relu_desc_,
                                       &alpha_, x_desc_, d_X,
                                       &beta_, y_desc_, d_y));

    check_cuda(cudaMemcpy(y_.values, d_y,
                          y_.rows * y_.columns * sizeof(float),
                          cudaMemcpyDeviceToHost));



    // free GPU memory
    check_cuda(cudaFree(d_X));
    check_cuda(cudaFree(d_y));

    return y_;
}

matrix<float> Relu::backward(matrix<float> in_gradients) {
    float *d_y;
    check_cuda(cudaMalloc(&d_y, y_.rows * y_.columns * sizeof(float)));
    check_cuda(cudaMemcpy(d_y, y_.values,
                          y_.rows * y_.columns * sizeof(float),
                          cudaMemcpyHostToDevice));

    float *d_dy;
    check_cuda(cudaMalloc(&d_dy, in_gradients.rows * in_gradients.columns * sizeof(float)));
    check_cuda(cudaMemcpy(d_dy, in_gradients.values,
                          in_gradients.rows * in_gradients.columns * sizeof(float),
                          cudaMemcpyHostToDevice));
    cudnnTensorDescriptor_t dy_desc;
    check_cudnn(cudnnCreateTensorDescriptor(&dy_desc));
    check_cudnn(cudnnSetTensor4dDescriptor(dy_desc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           1, 1, in_gradients.rows, in_gradients.columns));

    float *d_x;
    check_cuda(cudaMalloc(&d_x, x_.rows * x_.columns * sizeof(float)));
    check_cuda(cudaMemcpy(d_x, x_.values,
                          x_.rows * x_.columns * sizeof(float),
                          cudaMemcpyHostToDevice));

    float *d_dx;
    check_cuda(cudaMalloc(&d_dx, x_.rows * x_.columns * sizeof(float)));
    cudnnTensorDescriptor_t dx_desc;
    check_cudnn(cudnnCreateTensorDescriptor(&dx_desc));
    check_cudnn(cudnnSetTensor4dDescriptor(dx_desc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           1, 1, x_.rows, x_.columns));


    check_cudnn(cudnnActivationBackward(cuda_helper_->cudnn_handle,
                                        relu_desc_,
                                        &alpha_, y_desc_, d_y,
                                        dy_desc, d_dy,
                                        x_desc_, d_x,
                                        &beta_, dx_desc, d_dx));

    matrix<float> grad_input;
    grad_input.rows = x_.rows;
    grad_input.columns = x_.columns;
    grad_input.values = reinterpret_cast<float *>(malloc(grad_input.rows * grad_input.columns * sizeof(float)));
    check_cuda(cudaMemcpy(grad_input.values, d_dx,
                          grad_input.rows * grad_input.columns * sizeof(float),
                          cudaMemcpyDeviceToHost));

    return grad_input;
}

LogSoftmax::LogSoftmax(CudaHelper *helper) {
    cuda_helper_ = helper;
    alpha_ = 1.0;
    beta_ = 0.0;
}

matrix<float> LogSoftmax::forward(matrix<float> X) {
    // cudnn Tensor is row-major
    to_row_major(&X);

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

    y_.rows = X.rows;
    y_.columns = X.columns;
    y_.values = (float *) malloc(y_.rows * y_.columns * sizeof(float));
    for (int i = 0; i < y_.rows * y_.columns; ++i) {
        y_.values[i] = 0.0;
    }
    float *d_y;
    check_cuda(cudaMalloc(&d_y, y_.rows * y_.columns * sizeof(float)));
    check_cuda(cudaMemcpy(d_y, y_.values,
                          y_.rows * y_.columns * sizeof(float),
                          cudaMemcpyHostToDevice));
    cudnnTensorDescriptor_t y_desc;
    check_cudnn(cudnnCreateTensorDescriptor(&y_desc));
    check_cudnn(cudnnSetTensor4dDescriptor(y_desc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           y_.rows, 1, 1, y_.columns));

    check_cudnn(cudnnSoftmaxForward(cuda_helper_->cudnn_handle,
                                    CUDNN_SOFTMAX_LOG,
                                    CUDNN_SOFTMAX_MODE_INSTANCE,
                                    &alpha_, x_desc, d_X,
                                    &beta_, y_desc, d_y));

    check_cuda(cudaMemcpy(y_.values, d_y,
                          y_.rows * y_.columns * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // free GPU memory
    check_cuda(cudaFree(d_X));
    check_cuda(cudaFree(d_y));

    to_column_major(&y_);

    return y_;
}

matrix<float> LogSoftmax::backward(matrix<float> in_gradients) {
    cudnnTensorDescriptor_t y_desc;
    float *d_y;
    check_cuda(cudaMalloc(&d_y, y_.rows * y_.columns * sizeof(float)));
    check_cuda(cudaMemcpy(d_y, y_.values,
                          y_.rows * y_.columns * sizeof(float),
                          cudaMemcpyHostToDevice));
    check_cudnn(cudnnCreateTensorDescriptor(&y_desc));
    check_cudnn(cudnnSetTensor4dDescriptor(y_desc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           y_.rows, 1, 1, y_.columns));

    cudnnTensorDescriptor_t dy_desc;
    float *d_dy;
    check_cuda(cudaMalloc(&d_dy, y_.rows * y_.columns * sizeof(float)));
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
    check_cuda(cudaMalloc(&d_dx, y_.rows * y_.columns * sizeof(float)));
    check_cudnn(cudnnCreateTensorDescriptor(&dx_desc));
    check_cudnn(cudnnSetTensor4dDescriptor(dx_desc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           y_.rows, 1, 1, y_.columns));

    check_cudnn(cudnnSoftmaxBackward(cuda_helper_->cudnn_handle,
                                     CUDNN_SOFTMAX_LOG,
                                     CUDNN_SOFTMAX_MODE_INSTANCE,
                                     &alpha_, y_desc, d_y,
                                     dy_desc, d_dy,
                                     &beta_, dx_desc, d_dx));

    matrix<float> gradients;
    gradients.rows = y_.rows;
    gradients.columns = y_.columns;
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
