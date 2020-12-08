// Copyright 2020 Marcel Wagenländer

#include "dropout.hpp"
#include "cuda_helper.hpp"
#include "tensors.hpp"

#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cudnn_v7.h>
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
    forward(x, &y_);

    return &y_;
}

void Dropout::forward(Matrix<float> *x, Matrix<float> *y) {
    to_row_major_inplace(x);

    seed_ = rand();
    check_cudnn(cudnnDropoutGetStatesSize(cuda_helper_->cudnn_handle, &state_size_));
    void *d_states;
    check_cuda(cudaMalloc(&d_states, state_size_));
    cudnnDropoutDescriptor_t dropout_desc;
    check_cudnn(cudnnCreateDropoutDescriptor(&dropout_desc));
    check_cudnn(cudnnSetDropoutDescriptor(dropout_desc,
                                          cuda_helper_->cudnn_handle, probability_,
                                          d_states, state_size_, seed_));

    cudnnTensorDescriptor_t x_descr;
    check_cudnn(cudnnCreateTensorDescriptor(&x_descr));
    check_cudnn(cudnnSetTensor4dDescriptor(x_descr,
                                           CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           x->num_rows_, 1, 1, x->num_columns_));
    cudnnTensorDescriptor_t y_descr;
    check_cudnn(cudnnCreateTensorDescriptor(&y_descr));
    check_cudnn(cudnnSetTensor4dDescriptor(y_descr,
                                           CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           y->num_rows_, 1, 1, y->num_columns_));

    void *d_x;
    check_cuda(cudaMalloc(&d_x, x->size_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_x, x->values_, x->size_ * sizeof(float),
                          cudaMemcpyHostToDevice));
    void *d_y;
    check_cuda(cudaMalloc(&d_y, y->size_ * sizeof(float)));

    void *d_reserve_space;
    check_cudnn(cudnnDropoutGetReserveSpaceSize(x_descr, &reserve_space_size_));
    check_cuda(cudaMalloc(&d_reserve_space, reserve_space_size_));
    check_cudnn(cudnnDropoutForward(cuda_helper_->cudnn_handle,
                                    dropout_desc, x_descr, d_x,
                                    y_descr, d_y,
                                    d_reserve_space, reserve_space_size_));

    check_cuda(cudaMemcpy(y->values_, d_y, y->size_ * sizeof(float),
                          cudaMemcpyDeviceToHost));

    y->is_row_major_ = true;

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
}

Matrix<float> *Dropout::backward(Matrix<float> *incoming_gradients) {
    backward(incoming_gradients, &y_, &gradients_);

    return &gradients_;
}

void Dropout::backward(Matrix<float> *incoming_gradients, Matrix<float> *y, Matrix<float> *gradients) {
    if (y->num_rows_ != incoming_gradients->num_rows_ || y->num_columns_ != incoming_gradients->num_columns_) {
        throw "Matrix shapes are unequal";
    }
    to_row_major_inplace(incoming_gradients);
    to_row_major_inplace(y);

    void *d_states;
    check_cuda(cudaMalloc(&d_states, state_size_));
    cudnnDropoutDescriptor_t dropout_desc;
    check_cudnn(cudnnCreateDropoutDescriptor(&dropout_desc));
    check_cudnn(cudnnSetDropoutDescriptor(dropout_desc,
                                          cuda_helper_->cudnn_handle, probability_,
                                          d_states, state_size_, seed_));

    cudnnTensorDescriptor_t dy_desc;
    check_cudnn(cudnnCreateTensorDescriptor(&dy_desc));
    check_cudnn(cudnnSetTensor4dDescriptor(dy_desc,
                                           CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           incoming_gradients->num_rows_, 1, 1, incoming_gradients->num_columns_));
    float *d_dy;
    check_cuda(cudaMalloc(&d_dy, incoming_gradients->size_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_dy, incoming_gradients->values_,
                          incoming_gradients->size_,
                          cudaMemcpyHostToDevice));
    cudnnTensorDescriptor_t dx_desc;
    check_cudnn(cudnnCreateTensorDescriptor(&dx_desc));
    check_cudnn(cudnnSetTensor4dDescriptor(dx_desc,
                                           CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           incoming_gradients->num_rows_, 1, 1, incoming_gradients->num_columns_));
    float *d_dx;
    check_cuda(cudaMalloc(&d_dx, incoming_gradients->size_ * sizeof(float)));

    void *d_reserve_space;
    check_cuda(cudaMalloc(&d_reserve_space, reserve_space_size_));
    check_cuda(cudaMemcpy(d_reserve_space, reserve_space_,
                          reserve_space_size_,
                          cudaMemcpyHostToDevice));

    // It is expected that reserveSpace was populated during a call to cudnnDropoutForward and has not been changed
    check_cudnn(cudnnDropoutBackward(cuda_helper_->cudnn_handle,
                                     dropout_desc,
                                     dy_desc, d_dy,
                                     dx_desc, d_dx,
                                     d_reserve_space, reserve_space_size_));

    check_cuda(cudaMemcpy(gradients->values_, d_dx,
                          gradients->size_ * sizeof(float),
                          cudaMemcpyDeviceToHost));

    //clean-up
    check_cuda(cudaFree(d_dy));
    check_cuda(cudaFree(d_dx));
    check_cuda(cudaFree(d_reserve_space));
}

DropoutChunked::DropoutChunked() {}

DropoutChunked::DropoutChunked(CudaHelper *helper, int chunk_size, int num_nodes, long num_features) {
    set(helper, chunk_size, num_nodes, num_features);
}

void DropoutChunked::set(CudaHelper *helper, long chunk_size, long num_nodes, long num_features) {
    cuda_helper_ = helper;
    chunk_size_ = chunk_size;
    num_chunks_ = ceil((float) num_nodes / (float) chunk_size_);

    if (num_chunks_ * chunk_size_ > num_nodes) {
        last_chunk_size_ = num_nodes - (num_chunks_ - 1) * chunk_size_;
    } else {
        last_chunk_size_ = chunk_size_;
    }

    dropout_layers_ = std::vector<Dropout>(num_chunks_);
    y_ = std::vector<Matrix<float>>(num_chunks_);
    gradients_ = std::vector<Matrix<float>>(num_chunks_);
    long current_chunk_size = chunk_size_;
    for (int i = 0; i < num_chunks_; ++i) {
        if (i == num_chunks_ - 1) {
            current_chunk_size = last_chunk_size_;
        }
        dropout_layers_.at(i).set(cuda_helper_, current_chunk_size, num_features);
        y_.at(i).set(current_chunk_size, num_features, true);
        gradients_.at(i).set(current_chunk_size, num_features, true);
    }
}

std::vector<Matrix<float>> *DropoutChunked::forward(std::vector<Matrix<float>> *x) {
    for (int i = 0; i < x->size(); ++i) {
        to_row_major_inplace(&x->at(i));
    }

    for (int i = 0; i < num_chunks_; ++i) {
        dropout_layers_[i].forward(&x->at(i), &y_.at(i));
    }

    return &y_;
}

std::vector<Matrix<float>> *DropoutChunked::backward(std::vector<Matrix<float>> *incoming_gradients) {
    for (int i = 0; i < y_.size(); ++i) {
        to_row_major_inplace(&incoming_gradients->at(i));
        to_row_major_inplace(&y_.at(i));
    }

    for (int i = 0; i < num_chunks_; ++i) {
        dropout_layers_[i].backward(&incoming_gradients->at(i), &y_.at(i), &gradients_.at(i));
    }

    return &gradients_;
}
