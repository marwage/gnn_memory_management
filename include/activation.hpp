// Copyright 2020 Marcel Wagenländer

#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "cuda_helper.hpp"
#include "tensors.hpp"

#include <vector>


class ReluParent {
public:
    virtual Matrix<float>* forward(Matrix<float> *x) = 0;
    virtual Matrix<float>* backward(Matrix<float> *in_gradients) = 0;
};

class Relu : public ReluParent {
private:
    float alpha_;
    float beta_;
    CudaHelper *cuda_helper_;
    cudnnActivationDescriptor_t relu_desc_;
    cudnnTensorDescriptor_t y_desc_;
    cudnnTensorDescriptor_t x_desc_;
    Matrix<float> y_;
    Matrix<float> *x_;
    Matrix<float> gradients_;


public:
    Relu();
    Relu(CudaHelper *helper, long num_nodes, long num_features);
    Matrix<float>* forward(Matrix<float> *x);
    Matrix<float>* backward(Matrix<float> *in_gradients);
};

class ReluChunked : public ReluParent {
private:
    std::vector<Relu> relu_layers_;
    CudaHelper *cuda_helper_;
    long chunk_size_;
    long last_chunk_size_;
    long num_chunks_;
    std::vector<Matrix<float>> x_chunks_;
    Matrix<float> y_;
    Matrix<float> gradients_;

public:
    ReluChunked(CudaHelper *helper, long chunk_size, long num_nodes, long num_features);
    Matrix<float>* forward(Matrix<float> *x);
    Matrix<float>* backward(Matrix<float> *in_gradients);
};

class LogSoftmaxParent {
public:
    virtual Matrix<float>* forward(Matrix<float> *x) = 0;
    virtual Matrix<float>* backward(Matrix<float> *in_gradients) = 0;
};

class LogSoftmax : public LogSoftmaxParent {
private:
    CudaHelper *cuda_helper_;
    float alpha_;
    float beta_;
    Matrix<float> y_;
    Matrix<float> gradients_;

public:
    LogSoftmax();
    LogSoftmax(CudaHelper *helper, long num_nodes, long num_features);
    Matrix<float>* forward(Matrix<float> *x);
    Matrix<float>* backward(Matrix<float> *in_gradients);
};

class LogSoftmaxChunked : public LogSoftmaxParent {
private:
    std::vector<LogSoftmax> log_softmax_layers_;
    CudaHelper *cuda_helper_;
    long chunk_size_;
    long last_chunk_size_;
    long num_chunks_;
    Matrix<float> y_;
    Matrix<float> gradients_;

public:
    LogSoftmaxChunked(CudaHelper *helper, long chunk_size, long num_nodes, long num_features);
    Matrix<float>* forward(Matrix<float> *x);
    Matrix<float>* backward(Matrix<float> *in_gradients);
};

#endif
