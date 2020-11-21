// Copyright 2020 Marcel Wagenländer
#include "activation.hpp"
#include "cuda_helper.hpp"

#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <limits>


Relu::Relu() {}

Relu::Relu(CudaHelper *helper, long num_nodes, long num_features) {
    cuda_helper_ = helper;
    alpha_ = 1.0;
    beta_ = 0.0;

    y_ = new_float_matrix(num_nodes, num_features, true);
    gradients_ = new_float_matrix(num_nodes, num_features, true);
}

matrix<float> Relu::forward(matrix<float> x) {
    to_row_major_inplace(&x);
    to_row_major_inplace(&y_);
    if (y_.rows != x.rows || y_.columns != x.columns) {
        throw "Matrix shapes are unequal";
    }
    x_ = x;

    float *d_x;
    check_cuda(cudaMalloc(&d_x, x.rows * x.columns * sizeof(float)));
    check_cuda(cudaMemcpy(d_x, x.values,
                          x.rows * x.columns * sizeof(float),
                          cudaMemcpyHostToDevice));
    check_cudnn(cudnnCreateTensorDescriptor(&x_desc_));
    check_cudnn(cudnnSetTensor4dDescriptor(x_desc_,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           x.rows, 1, 1, x.columns));

    float *d_y;
    check_cuda(cudaMalloc(&d_y, y_.rows * y_.columns * sizeof(float)));
    check_cuda(cudaMemcpy(d_y, y_.values,
                          y_.rows * y_.columns * sizeof(float),
                          cudaMemcpyHostToDevice));
    check_cudnn(cudnnCreateTensorDescriptor(&y_desc_));
    check_cudnn(cudnnSetTensor4dDescriptor(y_desc_,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           y_.rows, 1, 1, y_.columns));

    check_cudnn(cudnnCreateActivationDescriptor(&relu_desc_));
    double coef = std::numeric_limits<double>::max();
    check_cudnn(cudnnSetActivationDescriptor(relu_desc_,
                                             CUDNN_ACTIVATION_RELU,
                                             CUDNN_PROPAGATE_NAN,
                                             coef));

    check_cudnn(cudnnActivationForward(cuda_helper_->cudnn_handle,
                                       relu_desc_,
                                       &alpha_, x_desc_, d_x,
                                       &beta_, y_desc_, d_y));

    check_cuda(cudaMemcpy(y_.values, d_y,
                          y_.rows * y_.columns * sizeof(float),
                          cudaMemcpyDeviceToHost));


    // free GPU memory
    check_cuda(cudaFree(d_x));
    check_cuda(cudaFree(d_y));

    return y_;
}

matrix<float> Relu::backward(matrix<float> in_gradients) {
    to_row_major_inplace(&in_gradients);
    to_row_major_inplace(&y_);
    if (y_.rows != in_gradients.rows || y_.columns != in_gradients.columns) {
        throw "Matrix shapes are unequal";
    }

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
                                           in_gradients.rows, 1, 1, in_gradients.columns));

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
                                           x_.rows, 1, 1, x_.columns));


    check_cudnn(cudnnActivationBackward(cuda_helper_->cudnn_handle,
                                        relu_desc_,
                                        &alpha_, y_desc_, d_y,
                                        dy_desc, d_dy,
                                        x_desc_, d_x,
                                        &beta_, dx_desc, d_dx));

    check_cuda(cudaMemcpy(gradients_.values, d_dx,
                          gradients_.rows * gradients_.columns * sizeof(float),
                          cudaMemcpyDeviceToHost));

    check_cuda(cudaFree(d_x));
    check_cuda(cudaFree(d_dx));
    check_cuda(cudaFree(d_y));
    check_cuda(cudaFree(d_dy));

    return gradients_;
}

LogSoftmax::LogSoftmax() {}

LogSoftmax::LogSoftmax(CudaHelper *helper, long num_nodes, long num_features) {
    cuda_helper_ = helper;
    alpha_ = 1.0;
    beta_ = 0.0;

    y_ = new_float_matrix(num_nodes, num_features, true);
    gradients_ = new_float_matrix(num_nodes, num_features, true);
}

matrix<float> LogSoftmax::forward(matrix<float> x) {
    to_row_major_inplace(&x);
    to_row_major_inplace(&y_);
    if (y_.rows != x.rows || y_.columns != x.columns) {
        throw "Matrix shapes are unequal";
    }

    float *d_x;
    check_cuda(cudaMalloc(&d_x, x.rows * x.columns * sizeof(float)));
    check_cuda(cudaMemcpy(d_x, x.values,
                          x.rows * x.columns * sizeof(float),
                          cudaMemcpyHostToDevice));
    cudnnTensorDescriptor_t x_desc;
    check_cudnn(cudnnCreateTensorDescriptor(&x_desc));
    check_cudnn(cudnnSetTensor4dDescriptor(x_desc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           x.rows, 1, 1, x.columns));

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
                                    &alpha_, x_desc, d_x,
                                    &beta_, y_desc, d_y));

    check_cuda(cudaMemcpy(y_.values, d_y,
                          y_.rows * y_.columns * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // free GPU memory
    check_cuda(cudaFree(d_x));
    check_cuda(cudaFree(d_y));

    return y_;
}

matrix<float> LogSoftmax::backward(matrix<float> in_gradients) {
    to_row_major_inplace(&in_gradients);
    to_row_major_inplace(&y_);
    if (y_.rows != in_gradients.rows || y_.columns != in_gradients.columns) {
        throw "Matrix shapes are unequal";
    }

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

    check_cuda(cudaMemcpy(gradients_.values, d_dx,
                          gradients_.rows * gradients_.columns * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // free GPU memory
    check_cuda(cudaFree(d_y));
    check_cuda(cudaFree(d_dy));
    check_cuda(cudaFree(d_dx));

    return gradients_;
}

ReluChunked::ReluChunked(CudaHelper *helper, long chunk_size, long num_nodes, long num_features) {
    chunk_size_ = chunk_size;
    cuda_helper_ = helper;
    num_chunks_ = ceil((float) num_nodes / (float) chunk_size_);

    if (num_chunks_ * chunk_size_ > num_nodes) {
        last_chunk_size_ = num_nodes - (num_chunks_ - 1) * chunk_size_;
    } else {
        last_chunk_size_ = chunk_size_;
    }
    relu_layers_ = std::vector<Relu>(num_chunks_);
    for (int i = 0; i < num_chunks_; ++i) {
        if (i == num_chunks_ - 1) {
            relu_layers_[i] = Relu(cuda_helper_, last_chunk_size_, num_features);
        } else {
            relu_layers_[i] = Relu(cuda_helper_, chunk_size_, num_features);
        }
    }

    y_ = new_float_matrix(num_nodes, num_features, true);
    gradients_ = new_float_matrix(num_nodes, num_features, true);
}

matrix<float> ReluChunked::forward(matrix<float> x) {
    to_row_major_inplace(&x);
    to_row_major_inplace(&y_);
    if (y_.rows != x.rows || y_.columns != x.columns) {
        throw "Matrix shapes are unequal";
    }

    matrix<float> X_chunk;
    X_chunk.rows = chunk_size_;
    X_chunk.columns = x.columns;
    matrix<float> Y_chunk;

    for (int i = 0; i < num_chunks_; ++i) {
        if (i == (num_chunks_ - 1)) {
            X_chunk.rows = last_chunk_size_;
        }
        X_chunk.values = &x.values[i * chunk_size_ * x.columns];

        Y_chunk = relu_layers_[i].forward(X_chunk);
        to_row_major_inplace(&Y_chunk);

        std::memcpy(&y_.values[i * chunk_size_ * Y_chunk.columns], Y_chunk.values, Y_chunk.rows * Y_chunk.columns * sizeof(float));
    }

    return y_;
}

matrix<float> ReluChunked::backward(matrix<float> in_gradients) {
    to_row_major_inplace(&in_gradients);
    if (y_.rows != in_gradients.rows || y_.columns != in_gradients.columns) {
        throw "Matrix shapes are unequal";
    }

    matrix<float> in_gradients_chunk;
    in_gradients_chunk.rows = chunk_size_;
    in_gradients_chunk.columns = in_gradients.columns;
    matrix<float> input_gradients_chunk;

    for (int i = 0; i < num_chunks_; ++i) {
        if (i == (num_chunks_ - 1)) {
            in_gradients_chunk.rows = last_chunk_size_;
        }
        in_gradients_chunk.values = &in_gradients.values[i * chunk_size_ * in_gradients.columns];

        input_gradients_chunk = relu_layers_[i].backward(in_gradients_chunk);
        to_row_major_inplace(&in_gradients_chunk);

        std::memcpy(&gradients_.values[i * chunk_size_ * input_gradients_chunk.columns], input_gradients_chunk.values, input_gradients_chunk.rows * input_gradients_chunk.columns * sizeof(float));
    }

    return gradients_;
}

LogSoftmaxChunked::LogSoftmaxChunked(CudaHelper *helper, long chunk_size, long num_nodes, long num_features) {
    chunk_size_ = chunk_size;
    cuda_helper_ = helper;
    num_chunks_ = ceil((float) num_nodes / (float) chunk_size_);

    if (num_chunks_ * chunk_size_ > num_nodes) {
        last_chunk_size_ = num_nodes - (num_chunks_ - 1) * chunk_size_;
    } else {
        last_chunk_size_ = chunk_size_;
    }

    log_softmax_layers_ = std::vector<LogSoftmax>(num_chunks_);
    for (int i = 0; i < num_chunks_; ++i) {
        if (i == num_chunks_ - 1) {
            log_softmax_layers_[i] = LogSoftmax(cuda_helper_, last_chunk_size_, num_features);
        } else {
            log_softmax_layers_[i] = LogSoftmax(cuda_helper_, chunk_size_, num_features);
        }
    }

    y_ = new_float_matrix(num_nodes, num_features, true);
    gradients_ = new_float_matrix(num_nodes, num_features, true);
}

matrix<float> LogSoftmaxChunked::forward(matrix<float> x) {
    to_row_major_inplace(&x);
    to_row_major_inplace(&y_);
    if (y_.rows != x.rows || y_.columns != x.columns) {
        throw "Matrix shapes are unequal";
    }

    matrix<float> X_chunk;
    matrix<float> Y_chunk;
    X_chunk.rows = chunk_size_;
    X_chunk.columns = x.columns;

    for (int i = 0; i < num_chunks_; ++i) {
        if (i == (num_chunks_ - 1)) {
            X_chunk.rows = last_chunk_size_;
        }
        X_chunk.values = &x.values[i * chunk_size_ * x.columns];

        Y_chunk = log_softmax_layers_[i].forward(X_chunk);
        to_row_major_inplace(&Y_chunk);

        std::memcpy(&y_.values[i * chunk_size_ * Y_chunk.columns], Y_chunk.values, Y_chunk.rows * Y_chunk.columns * sizeof(float));
    }

    return y_;
}

matrix<float> LogSoftmaxChunked::backward(matrix<float> in_gradients) {
    to_row_major_inplace(&in_gradients);
    if (y_.rows != in_gradients.rows || y_.columns != in_gradients.columns) {
        throw "Matrix shapes are unequal";
    }

    matrix<float> gradients_chunk;
    matrix<float> in_gradients_chunk;
    in_gradients_chunk.rows = chunk_size_;
    in_gradients_chunk.columns = in_gradients.columns;

    for (int i = 0; i < num_chunks_; ++i) {
        if (i == (num_chunks_ - 1)) {
            in_gradients_chunk.rows = last_chunk_size_;
        }
        in_gradients_chunk.values = &in_gradients.values[i * chunk_size_ * in_gradients.columns];

        gradients_chunk = log_softmax_layers_[i].backward(in_gradients_chunk);
        to_row_major_inplace(&gradients_chunk);

        std::memcpy(&gradients_.values[i * chunk_size_ * in_gradients.columns], gradients_chunk.values, gradients_chunk.rows * gradients_chunk.columns * sizeof(float));
    }

    return gradients_;
}
