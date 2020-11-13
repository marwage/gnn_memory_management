// Copyright 2020 Marcel Wagenländer

#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "cuda_helper.hpp"
#include "tensors.hpp"

#include <vector>

class Relu {
private:
    float alpha_;
    float beta_;
    CudaHelper *cuda_helper_;
    cudnnActivationDescriptor_t relu_desc_;
    cudnnTensorDescriptor_t y_desc_;
    matrix<float> y_;
    matrix<float> x_;
    cudnnTensorDescriptor_t x_desc_;

public:
    Relu();
    Relu(CudaHelper *helper);
    matrix<float> forward(matrix<float> X);
    matrix<float> backward(matrix<float> in_gradients);
};

class LogSoftmax {
private:
    float alpha_;
    float beta_;
    matrix<float> y_;
    CudaHelper *cuda_helper_;

public:
    LogSoftmax();
    LogSoftmax(CudaHelper *helper);
    matrix<float> forward(matrix<float> X);
    matrix<float> backward(matrix<float> in_gradients);
};

class ReluChunked {
private:
    Relu relu_layer_;
    int chunk_size_;
    int last_chunk_size_;
    int num_chunks_;

public:
    ReluChunked(CudaHelper *helper, int chunk_size, int num_nodes);
    matrix<float> forward(matrix<float> X);
    matrix<float> backward(matrix<float> in_gradients);
};

class LogSoftmaxChunked {
private:
    std::vector<LogSoftmax> log_softmax_layers_;
    CudaHelper *cuda_helper_;
    int chunk_size_;
    int last_chunk_size_;
    int num_chunks_;

public:
    LogSoftmaxChunked(CudaHelper *helper, int chunk_size, int num_nodes);
    matrix<float> forward(matrix<float> X);
    matrix<float> backward(matrix<float> in_gradients);
};

#endif
