// Copyright 2020 Marcel Wagenländer

#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "cuda_helper.hpp"
#include "tensors.hpp"

#include <vector>


class ReluParent {
public:
    virtual matrix<float> forward(matrix<float> X) = 0;
    virtual matrix<float> backward(matrix<float> in_gradients) = 0;
};

class Relu : public ReluParent {
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

class ReluChunked : public ReluParent {
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

class LogSoftmaxParent {
public:
    virtual matrix<float> forward(matrix<float> X) = 0;
    virtual matrix<float> backward(matrix<float> in_gradients) = 0;
};

class LogSoftmax : public LogSoftmaxParent {
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

class LogSoftmaxChunked : public LogSoftmaxParent {
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
