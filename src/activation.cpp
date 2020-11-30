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

    y_.set(num_nodes, num_features, true);
    gradients_.set(num_nodes, num_features, true);
}

Matrix<float> *Relu::forward(Matrix<float> *x) {
    to_row_major_inplace(x);
    if (y_.num_rows_ != x->num_rows_ || y_.num_columns_ != x->num_columns_) {
        throw "Matrix shapes are unequal";
    }
    x_ = x;

    float *d_x;
    check_cuda(cudaMalloc(&d_x, x->size_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_x, x->values_, x->size_ * sizeof(float),
                          cudaMemcpyHostToDevice));
    check_cudnn(cudnnCreateTensorDescriptor(&x_desc_));
    check_cudnn(cudnnSetTensor4dDescriptor(x_desc_,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           x->num_rows_, 1, 1, x->num_columns_));

    float *d_y;
    check_cuda(cudaMalloc(&d_y, y_.size_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_y, y_.values_, y_.size_ * sizeof(float),
                          cudaMemcpyHostToDevice));
    check_cudnn(cudnnCreateTensorDescriptor(&y_desc_));
    check_cudnn(cudnnSetTensor4dDescriptor(y_desc_,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           y_.num_rows_, 1, 1, y_.num_columns_));

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

    check_cuda(cudaMemcpy(y_.values_, d_y, y_.size_ * sizeof(float),
                          cudaMemcpyDeviceToHost));

    y_.is_row_major_ = true;

    // free GPU memory
    check_cuda(cudaFree(d_x));
    check_cuda(cudaFree(d_y));

    return &y_;
}

Matrix<float> *Relu::backward(Matrix<float> *in_gradients) {
    to_row_major_inplace(in_gradients);
    to_row_major_inplace(&y_);
    to_row_major_inplace(x_);
    if (y_.num_rows_ != in_gradients->num_rows_ || y_.num_columns_ != in_gradients->num_columns_) {
        throw "Matrix shapes are unequal";
    }

    float *d_y;
    check_cuda(cudaMalloc(&d_y, y_.size_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_y, y_.values_,
                          y_.size_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    float *d_dy;
    check_cuda(cudaMalloc(&d_dy, in_gradients->size_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_dy, in_gradients->values_, in_gradients->size_ * sizeof(float),
                          cudaMemcpyHostToDevice));
    cudnnTensorDescriptor_t dy_desc;
    check_cudnn(cudnnCreateTensorDescriptor(&dy_desc));
    check_cudnn(cudnnSetTensor4dDescriptor(dy_desc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           in_gradients->num_rows_, 1, 1, in_gradients->num_columns_));

    float *d_x;
    check_cuda(cudaMalloc(&d_x, x_->size_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_x, x_->values_, x_->size_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    float *d_dx;
    check_cuda(cudaMalloc(&d_dx, x_->size_ * sizeof(float)));
    cudnnTensorDescriptor_t dx_desc;
    check_cudnn(cudnnCreateTensorDescriptor(&dx_desc));
    check_cudnn(cudnnSetTensor4dDescriptor(dx_desc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           x_->num_rows_, 1, 1, x_->num_columns_));


    check_cudnn(cudnnActivationBackward(cuda_helper_->cudnn_handle,
                                        relu_desc_,
                                        &alpha_, y_desc_, d_y,
                                        dy_desc, d_dy,
                                        x_desc_, d_x,
                                        &beta_, dx_desc, d_dx));

    check_cuda(cudaMemcpy(gradients_.values_, d_dx,
                          gradients_.size_ * sizeof(float),
                          cudaMemcpyDeviceToHost));

    gradients_.is_row_major_ = true;

    check_cuda(cudaFree(d_x));
    check_cuda(cudaFree(d_dx));
    check_cuda(cudaFree(d_y));
    check_cuda(cudaFree(d_dy));

    return &gradients_;
}

LogSoftmax::LogSoftmax() {}

LogSoftmax::LogSoftmax(CudaHelper *helper, long num_nodes, long num_features) {
    cuda_helper_ = helper;
    alpha_ = 1.0;
    beta_ = 0.0;

    y_.set(num_nodes, num_features, true);
    gradients_.set(num_nodes, num_features, true);
}

Matrix<float> *LogSoftmax::forward(Matrix<float> *x) {
    to_row_major_inplace(x);
    if (y_.num_rows_ != x->num_rows_ || y_.num_columns_ != x->num_columns_) {
        throw "Matrix shapes are unequal";
    }

    float *d_x;
    check_cuda(cudaMalloc(&d_x, x->size_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_x, x->values_, x->size_ * sizeof(float),
                          cudaMemcpyHostToDevice));
    cudnnTensorDescriptor_t x_desc;
    check_cudnn(cudnnCreateTensorDescriptor(&x_desc));
    check_cudnn(cudnnSetTensor4dDescriptor(x_desc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           x->num_rows_, 1, 1, x->num_columns_));

    float *d_y;
    check_cuda(cudaMalloc(&d_y, y_.size_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_y, y_.values_, y_.size_ * sizeof(float),
                          cudaMemcpyHostToDevice));
    cudnnTensorDescriptor_t y_desc;
    check_cudnn(cudnnCreateTensorDescriptor(&y_desc));
    check_cudnn(cudnnSetTensor4dDescriptor(y_desc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           y_.num_rows_, 1, 1, y_.num_columns_));

    check_cudnn(cudnnSoftmaxForward(cuda_helper_->cudnn_handle,
                                    CUDNN_SOFTMAX_LOG,
                                    CUDNN_SOFTMAX_MODE_INSTANCE,
                                    &alpha_, x_desc, d_x,
                                    &beta_, y_desc, d_y));

    check_cuda(cudaMemcpy(y_.values_, d_y,
                          y_.num_rows_ * y_.num_columns_ * sizeof(float),
                          cudaMemcpyDeviceToHost));

    y_.is_row_major_ = true;

    // free GPU memory
    check_cuda(cudaFree(d_x));
    check_cuda(cudaFree(d_y));

    return &y_;
}

Matrix<float> *LogSoftmax::backward(Matrix<float> *in_gradients) {
    to_row_major_inplace(in_gradients);
    to_row_major_inplace(&y_);
    if (y_.num_rows_ != in_gradients->num_rows_ || y_.num_columns_ != in_gradients->num_columns_) {
        throw "Matrix shapes are unequal";
    }

    cudnnTensorDescriptor_t y_desc;
    float *d_y;
    check_cuda(cudaMalloc(&d_y, y_.size_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_y, y_.values_, y_.size_ * sizeof(float),
                          cudaMemcpyHostToDevice));
    check_cudnn(cudnnCreateTensorDescriptor(&y_desc));
    check_cudnn(cudnnSetTensor4dDescriptor(y_desc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           y_.num_rows_, 1, 1, y_.num_columns_));

    cudnnTensorDescriptor_t dy_desc;
    float *d_dy;
    check_cuda(cudaMalloc(&d_dy, in_gradients->num_rows_ * in_gradients->num_columns_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_dy, in_gradients->values_,
                          in_gradients->num_rows_ * in_gradients->num_columns_ * sizeof(float),
                          cudaMemcpyHostToDevice));
    check_cudnn(cudnnCreateTensorDescriptor(&dy_desc));
    check_cudnn(cudnnSetTensor4dDescriptor(dy_desc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           in_gradients->num_rows_, 1, 1, in_gradients->num_columns_));

    cudnnTensorDescriptor_t dx_desc;
    float *d_dx;
    check_cuda(cudaMalloc(&d_dx, y_.size_ * sizeof(float)));
    check_cudnn(cudnnCreateTensorDescriptor(&dx_desc));
    check_cudnn(cudnnSetTensor4dDescriptor(dx_desc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           y_.num_rows_, 1, 1, y_.num_columns_));

    check_cudnn(cudnnSoftmaxBackward(cuda_helper_->cudnn_handle,
                                     CUDNN_SOFTMAX_LOG,
                                     CUDNN_SOFTMAX_MODE_INSTANCE,
                                     &alpha_, y_desc, d_y,
                                     dy_desc, d_dy,
                                     &beta_, dx_desc, d_dx));

    check_cuda(cudaMemcpy(gradients_.values_, d_dx, gradients_.size_ * sizeof(float),
                          cudaMemcpyDeviceToHost));

    gradients_.is_row_major_ = true;

    // free GPU memory
    check_cuda(cudaFree(d_y));
    check_cuda(cudaFree(d_dy));
    check_cuda(cudaFree(d_dx));

    return &gradients_;
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
    x_chunks_ = std::vector<Matrix<float>>(num_chunks_);
    for (int i = 0; i < num_chunks_; ++i) {
        if (i == num_chunks_ - 1) {
            relu_layers_[i] = Relu(cuda_helper_, last_chunk_size_, num_features);
            x_chunks_[i].set(last_chunk_size_, num_features, true);
        } else {
            relu_layers_[i] = Relu(cuda_helper_, chunk_size_, num_features);
            x_chunks_[i].set(chunk_size, num_features, true);
        }
    }

    y_.set(num_nodes, num_features, true);
    gradients_.set(num_nodes, num_features, true);
}

Matrix<float> *ReluChunked::forward(Matrix<float> *x) {
    to_row_major_inplace(x);
    if (y_.num_rows_ != x->num_rows_ || y_.num_columns_ != x->num_columns_) {
        throw "Matrix shapes are unequal";
    }

    Matrix<float> *y_chunk;
    for (int i = 0; i < num_chunks_; ++i) {
        x_chunks_[i].values_ = &x->values_[i * chunk_size_ * x->num_columns_];

        y_chunk = relu_layers_[i].forward(&x_chunks_[i]);

        to_row_major_inplace(y_chunk);
        std::memcpy(&y_.values_[i * chunk_size_ * y_chunk->num_columns_], y_chunk->values_, y_chunk->num_rows_ * y_chunk->num_columns_ * sizeof(float));
    }

    y_.is_row_major_ = true;

    return &y_;
}

Matrix<float> *ReluChunked::backward(Matrix<float> *in_gradients) {
    to_row_major_inplace(in_gradients);
    if (y_.num_rows_ != in_gradients->num_rows_ || y_.num_columns_ != in_gradients->num_columns_) {
        throw "Matrix shapes are unequal";
    }

    Matrix<float> in_gradients_chunk;
    Matrix<float> *input_gradients_chunk;
    long current_chunk_size = chunk_size_;
    for (int i = 0; i < num_chunks_; ++i) {
        if (i == (num_chunks_ - 1)) {
            current_chunk_size = last_chunk_size_;
        }
        in_gradients_chunk.set(current_chunk_size, in_gradients->num_columns_,
                               &in_gradients->values_[i * chunk_size_ * in_gradients->num_columns_],
                               in_gradients->is_row_major_, false);

        input_gradients_chunk = relu_layers_[i].backward(&in_gradients_chunk);

        to_row_major_inplace(input_gradients_chunk);
        std::memcpy(&gradients_.values_[i * chunk_size_ * in_gradients->num_columns_],
                    input_gradients_chunk->values_,
                    input_gradients_chunk->num_rows_ * input_gradients_chunk->num_columns_ * sizeof(float));
    }

    gradients_.is_row_major_ = true;

    return &gradients_;
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

    y_.set(num_nodes, num_features, true);
    gradients_.set(num_nodes, num_features, true);
}

Matrix<float> *LogSoftmaxChunked::forward(Matrix<float> *x) {
    to_row_major_inplace(x);
    if (y_.num_rows_ != x->num_rows_ || y_.num_columns_ != x->num_columns_) {
        throw "Matrix shapes are unequal";
    }

    Matrix<float> x_chunk;
    Matrix<float> *y_chunk;
    long current_chunk_size = chunk_size_;
    for (int i = 0; i < num_chunks_; ++i) {
        if (i == (num_chunks_ - 1)) {
            current_chunk_size = last_chunk_size_;
        }
        x_chunk.set(current_chunk_size, x->num_columns_,
                    &x->values_[i * chunk_size_ * x->num_columns_],
                    x->is_row_major_, false);

        y_chunk = log_softmax_layers_[i].forward(&x_chunk);

        to_row_major_inplace(y_chunk);
        std::memcpy(&y_.values_[i * chunk_size_ * y_chunk->num_columns_], y_chunk->values_,
                    y_chunk->num_rows_ * y_chunk->num_columns_ * sizeof(float));
    }

    y_.is_row_major_ = true;

    return &y_;
}

Matrix<float> *LogSoftmaxChunked::backward(Matrix<float> *in_gradients) {
    to_row_major_inplace(in_gradients);
    if (y_.num_rows_ != in_gradients->num_rows_ || y_.num_columns_ != in_gradients->num_columns_) {
        throw "Matrix shapes are unequal";
    }

    Matrix<float> *gradients_chunk;
    Matrix<float> in_gradients_chunk;
    long current_chunk_size = chunk_size_;
    for (int i = 0; i < num_chunks_; ++i) {
        if (i == (num_chunks_ - 1)) {
            current_chunk_size = last_chunk_size_;
        }
        in_gradients_chunk.set(current_chunk_size, in_gradients->num_columns_,
                               &in_gradients->values_[i * chunk_size_ * in_gradients->num_columns_],
                               in_gradients->is_row_major_, false);

        gradients_chunk = log_softmax_layers_[i].backward(&in_gradients_chunk);

        to_row_major_inplace(gradients_chunk);
        std::memcpy(&gradients_.values_[i * chunk_size_ * in_gradients->num_columns_], gradients_chunk->values_,
                    gradients_chunk->num_rows_ * gradients_chunk->num_columns_ * sizeof(float));
    }

    gradients_.is_row_major_ = true;

    return &gradients_;
}
