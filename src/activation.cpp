// Copyright 2020 Marcel Wagenländer

#include <cuda_runtime.h>
#include <cudnn.h>
#include <limits>
#include <cmath>
#include <cstring>

#include "activation.hpp"
#include "cuda_helper.hpp"


Relu::Relu() {}

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

LogSoftmax::LogSoftmax() {}

LogSoftmax::LogSoftmax(CudaHelper *helper) {
    cuda_helper_ = helper;
    alpha_ = 1.0;
    beta_ = 0.0;
}

matrix<float> LogSoftmax::forward(matrix<float> X) {
    // cudnn Tensor is row-major
    matrix<float> X_row = to_row_major(&X);

    float *d_X;
    check_cuda(cudaMalloc(&d_X, X_row.rows * X_row.columns * sizeof(float)));
    check_cuda(cudaMemcpy(d_X, X_row.values,
                          X_row.rows * X_row.columns * sizeof(float),
                          cudaMemcpyHostToDevice));
    cudnnTensorDescriptor_t x_desc;
    check_cudnn(cudnnCreateTensorDescriptor(&x_desc));
    check_cudnn(cudnnSetTensor4dDescriptor(x_desc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           X.rows, 1, 1, X.columns));

    y_.rows = X_row.rows;
    y_.columns = X_row.columns;
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

    to_column_major_inplace(&y_);

    return y_;
}

matrix<float> LogSoftmax::backward(matrix<float> in_gradients) {
    to_row_major(&in_gradients);
    to_row_major(&y_);

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

    to_column_major(&gradients);

    // free GPU memory
    check_cuda(cudaFree(d_y));
    check_cuda(cudaFree(d_dy));
    check_cuda(cudaFree(d_dx));

    return gradients;
}

ReluChunked::ReluChunked(CudaHelper *helper, int chunk_size) {
    chunk_size_ = chunk_size;
    relu_layer_ = Relu(helper);
}

matrix<float> ReluChunked::forward(matrix<float> X) {
    num_chunks_ = ceil((float) X.rows / (float) chunk_size_);

    if (num_chunks_ * chunk_size_ > X.rows) {
        last_chunk_size_ = X.rows - (num_chunks_ - 1) * chunk_size_;
    } else {
        last_chunk_size_ = chunk_size_;
    }

    matrix<float> X_row = to_row_major(&X);

    matrix<float> Y;
    Y.rows = X.rows;
    Y.columns = X.columns;
    Y.values = reinterpret_cast<float *>(malloc(Y.rows * Y.columns * sizeof(float)));
    matrix<float> X_chunk;
    matrix<float> Y_chunk;

    for (int i = 0; i < num_chunks_; ++i) {
        if (i == (num_chunks_ - 1)) {
            X_chunk.rows = last_chunk_size_;
        } else {
            X_chunk.rows = chunk_size_;
        }
        X_chunk.columns = X_row.columns;
        X_chunk.values = &X_row.values[i * chunk_size_];

        Y_chunk = relu_layer_.forward(X_chunk);

        std::memcpy(&Y.values[i * chunk_size_], Y_chunk.values, Y_chunk.rows * Y_chunk.columns * sizeof(float));
    }

    to_column_major(&Y);

    return Y;
}

matrix<float> ReluChunked::backward(matrix<float> in_gradients) {

}

LogSoftmaxChunked::LogSoftmaxChunked(CudaHelper *helper, int chunk_size) {
    chunk_size_ = chunk_size;
    cuda_helper_ = helper;
}

matrix<float> LogSoftmaxChunked::forward(matrix<float> X) {
    num_chunks_ = ceil((float) X.rows / (float) chunk_size_);
    log_softmax_layers_ = std::vector<LogSoftmax>(num_chunks_);

    if (num_chunks_ * chunk_size_ > X.rows) {
        last_chunk_size_ = X.rows - (num_chunks_ - 1) * chunk_size_;
    } else {
        last_chunk_size_ = chunk_size_;
    }

    matrix<float> X_row = to_row_major(&X);

    matrix<float> Y;
    Y.rows = X.rows;
    Y.columns = X.columns;
    Y.values = reinterpret_cast<float *>(malloc(Y.rows * Y.columns * sizeof(float)));
    matrix<float> X_chunk;
    matrix<float> Y_chunk;
    X_chunk.rows = chunk_size_;
    X_chunk.columns = X_row.columns;

    for (int i = 0; i < num_chunks_; ++i) {
        log_softmax_layers_[i] = LogSoftmax(cuda_helper_);
        if (i == (num_chunks_ - 1)) {
            X_chunk.rows = last_chunk_size_;
        }
        X_chunk.values = &X_row.values[i * chunk_size_ * X_row.columns];
        to_column_major_inplace(&X_chunk);

        Y_chunk = log_softmax_layers_[i].forward(X_chunk);
        to_row_major_inplace(&Y_chunk);

        std::memcpy(&Y.values[i * chunk_size_ * Y_chunk.columns], Y_chunk.values, Y_chunk.rows * Y_chunk.columns * sizeof(float));
    }

    to_column_major_inplace(&Y);

    return Y;
}

matrix<float> LogSoftmaxChunked::backward(matrix<float> in_gradients) {
    matrix<float> in_gradients_row = to_row_major(&in_gradients);

    matrix<float> gradients;
    gradients.rows = in_gradients_row.rows;
    gradients.columns = in_gradients_row.columns;
    gradients.values = reinterpret_cast<float *>(malloc(gradients.rows * gradients.columns * sizeof(float)));
    matrix<float> in_gradients_chunk;
    matrix<float> gradients_chunk;
    in_gradients_chunk.rows = chunk_size_;
    in_gradients_chunk.columns = in_gradients_row.columns;

    for (int i = 0; i < num_chunks_; ++i) {
        if (i == (num_chunks_ - 1)) {
            in_gradients_chunk.rows = last_chunk_size_;
        }
        in_gradients_chunk.values = &in_gradients_row.values[i * chunk_size_ * in_gradients_row.columns];
        to_column_major_inplace(&in_gradients_chunk);

        gradients_chunk = log_softmax_layers_[i].backward(in_gradients_chunk);
        to_row_major_inplace(&gradients_chunk);

        std::memcpy(&gradients.values[i * chunk_size_ * gradients_chunk.columns], gradients_chunk.values, gradients_chunk.rows * gradients_chunk.columns * sizeof(float));
    }

    to_column_major_inplace(&gradients);

    return gradients;
}
