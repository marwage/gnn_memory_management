// Copyright 2020 Marcel Wagenländer

#include "dropout.hpp"
#include "cuda_helper.hpp"
#include "tensors.hpp"

#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>


Dropout::Dropout() {}

Dropout::Dropout(CudaHelper *helper, long num_nodes, long num_features) {
    set(helper, num_nodes, num_features);
}

Dropout::~Dropout() {
    delete[] reserve_space_;
    delete[] states_;
}

void Dropout::set(CudaHelper *helper, long num_nodes, long num_features) {
    cuda_helper_ = helper;

    y_.set(num_nodes, num_features, true);
    gradients_.set(num_nodes, num_features, true);
}

Matrix<float> *Dropout::forward(Matrix<float> *x) {
    to_row_major_inplace(x);
    if (y_.num_rows_ != x->num_rows_ || y_.num_columns_ != x->num_columns_) {
        throw "Matrix shapes are unequal";
    }

    float probability = 0.2f;
    check_cudnn(cudnnDropoutGetStatesSize(cuda_helper_->cudnn_handle, &state_size_));
    void *d_states;
    check_cuda(cudaMalloc(&d_states, state_size_));
    unsigned long long seed = rand();
    check_cudnn(cudnnCreateDropoutDescriptor(&dropout_desc_));
    check_cudnn(cudnnSetDropoutDescriptor(dropout_desc_,
                                          cuda_helper_->cudnn_handle, probability,
                                          d_states, state_size_, seed));

    cudnnTensorDescriptor_t x_descr;
    check_cudnn(cudnnCreateTensorDescriptor(&x_descr));
    check_cudnn(cudnnSetTensor4dDescriptor(x_descr,
                                           CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           x->num_rows_, 1, 1, x->num_columns_));
    cudnnTensorDescriptor_t y_descr;
    check_cudnn(cudnnCreateTensorDescriptor(&y_descr));
    check_cudnn(cudnnSetTensor4dDescriptor(y_descr,
                                           CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           y_.num_rows_, 1, 1, y_.num_columns_));

    void *d_x;
    check_cuda(cudaMalloc(&d_x, x->size_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_x, x->values_, x->size_ * sizeof(float),
                          cudaMemcpyHostToDevice));
    void *d_y;
    check_cuda(cudaMalloc(&d_y, y_.size_ * sizeof(float)));

    void *d_reserve_space;
    check_cudnn(cudnnDropoutGetReserveSpaceSize(x_descr, &reserve_space_size_));
    check_cuda(cudaMalloc(&d_reserve_space, reserve_space_size_));
    check_cudnn(cudnnDropoutForward(cuda_helper_->cudnn_handle,
                                    dropout_desc_, x_descr, d_x,
                                    y_descr, d_y,
                                    d_reserve_space, reserve_space_size_));

    // DEBUGGING
    float val = y_.values_[5];

    check_cuda(cudaMemcpy(y_.values_, d_y, y_.size_ * sizeof(float),
                          cudaMemcpyDeviceToHost));

    y_.is_row_major_ = true;

    if (reserve_space_ == NULL) {
        reserve_space_ = new char[reserve_space_size_];
    }
    check_cuda(cudaMemcpy(reserve_space_, d_reserve_space,
                          reserve_space_size_,
                          cudaMemcpyDeviceToHost));

    if (states_ == NULL) {
        states_ = new char[state_size_];
    }
    check_cuda(cudaMemcpy(states_, d_states, state_size_,
                          cudaMemcpyDeviceToHost));

    check_cuda(cudaFree(d_states));
    check_cuda(cudaFree(d_reserve_space));
    check_cuda(cudaFree(d_x));
    check_cuda(cudaFree(d_y));

    return &y_;
}

Matrix<float> *Dropout::backward(Matrix<float> *in_gradients) {
    to_row_major_inplace(in_gradients);
    to_row_major_inplace(&y_);
    if (y_.num_rows_ != in_gradients->num_rows_ || y_.num_columns_ != in_gradients->num_columns_) {
        throw "Matrix shapes are unequal";
    }

    cudnnTensorDescriptor_t dy_desc;
    check_cudnn(cudnnCreateTensorDescriptor(&dy_desc));
    check_cudnn(cudnnSetTensor4dDescriptor(dy_desc,
                                           CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           in_gradients->num_rows_, 1, 1, in_gradients->num_columns_));
    float *d_dy;
    check_cuda(cudaMalloc(&d_dy, in_gradients->size_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_dy, in_gradients->values_,
                          in_gradients->size_,
                          cudaMemcpyHostToDevice));
    cudnnTensorDescriptor_t dx_desc;
    check_cudnn(cudnnCreateTensorDescriptor(&dx_desc));
    check_cudnn(cudnnSetTensor4dDescriptor(dx_desc,
                                           CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           in_gradients->num_rows_, 1, 1, in_gradients->num_columns_));
    float *d_dx;
    check_cuda(cudaMalloc(&d_dx, in_gradients->size_ * sizeof(float)));

    void *d_reserve_space;
    check_cuda(cudaMalloc(&d_reserve_space, reserve_space_size_));
    check_cuda(cudaMemcpy(d_reserve_space, reserve_space_,
                          reserve_space_size_,
                          cudaMemcpyHostToDevice));

    // It is expected that reserveSpace was populated during a call to cudnnDropoutForward and has not been changed
    check_cudnn(cudnnDropoutBackward(cuda_helper_->cudnn_handle,
                                     dropout_desc_,
                                     dy_desc, d_dy,
                                     dx_desc, d_dx,
                                     d_reserve_space, reserve_space_size_));

    check_cuda(cudaMemcpy(gradients_.values_, d_dx,
                          gradients_.size_ * sizeof(float),
                          cudaMemcpyDeviceToHost));

    gradients_.is_row_major_ = true;

    //clean-up
    check_cuda(cudaFree(d_dy));
    check_cuda(cudaFree(d_dx));
    check_cuda(cudaFree(d_reserve_space));

    return &gradients_;
}

DropoutChunked::DropoutChunked(CudaHelper *helper, int chunk_size, int num_nodes, long num_features) {
    cuda_helper_ = helper;
    chunk_size_ = chunk_size;
    num_chunks_ = ceil((float) num_nodes / (float) chunk_size_);

    if (num_chunks_ * chunk_size_ > num_nodes) {
        last_chunk_size_ = num_nodes - (num_chunks_ - 1) * chunk_size_;
    } else {
        last_chunk_size_ = chunk_size_;
    }

    dropout_layers_ = std::vector<Dropout>(num_chunks_);
    long current_chunk_size = chunk_size_;
    for (int i = 0; i < num_chunks_; ++i) {
        if (i == num_chunks_ - 1) {
            current_chunk_size = last_chunk_size_;
        }
        dropout_layers_[i].set(cuda_helper_, current_chunk_size, num_features);
    }

    y_.set(num_nodes, num_features, true);
    gradients_.set(num_nodes, num_features, true);
}

Matrix<float> *DropoutChunked::forward(Matrix<float> *x) {
    to_row_major_inplace(x);

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

        y_chunk = dropout_layers_[i].forward(&x_chunk);

        to_row_major_inplace(y_chunk);
        std::memcpy(&y_.values_[i * chunk_size_ * y_chunk->num_columns_], y_chunk->values_, y_chunk->size_ * sizeof(float));
    }

    y_.is_row_major_ = true;

    return &y_;
}

Matrix<float> *DropoutChunked::backward(Matrix<float> *in_gradients) {
    to_row_major_inplace(in_gradients);
    to_row_major_inplace(&y_);

    Matrix<float> in_gradients_chunk;
    Matrix<float> *gradients_chunk;
    long current_chunk_size = chunk_size_;
    for (int i = 0; i < num_chunks_; ++i) {
        if (i == (num_chunks_ - 1)) {
            current_chunk_size = last_chunk_size_;
        }
        in_gradients_chunk.set(current_chunk_size, in_gradients->num_columns_,
                               &in_gradients->values_[i * chunk_size_ * in_gradients->num_columns_],
                               in_gradients->is_row_major_, false);

        gradients_chunk = dropout_layers_[i].backward(&in_gradients_chunk);

        to_row_major_inplace(gradients_chunk);
        std::memcpy(&gradients_.values_[i * chunk_size_ * gradients_chunk->num_columns_], gradients_chunk->values_,
                    gradients_chunk->size_ * sizeof(float));
    }

    gradients_.is_row_major_ = true;

    return &gradients_;
}
