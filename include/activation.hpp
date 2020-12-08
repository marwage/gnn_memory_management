// Copyright 2020 Marcel Wagenländer

#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "cuda_helper.hpp"
#include "tensors.hpp"
#include "layer.hpp"

#include <vector>


class Relu : public Layer {
private:
    float alpha_;
    float beta_;
    CudaHelper *cuda_helper_ = NULL;
    cudnnActivationDescriptor_t relu_desc_;
    Matrix<float> *x_ = NULL;
    Matrix<float> y_;
    Matrix<float> gradients_;

public:
    Relu();
    Relu(CudaHelper *helper);
    Relu(CudaHelper *helper, long num_nodes, long num_features);
    void set(CudaHelper *helper);
    void set(CudaHelper *helper, long num_nodes, long num_features) override;
    Matrix<float> *forward(Matrix<float> *x) override;
    void forward(Matrix<float> *x, Matrix<float> *y);
    Matrix<float> *backward(Matrix<float> *incoming_gradients) override;
    void backward(Matrix<float> *incoming_gradients, Matrix<float> *x, Matrix<float> *y, Matrix<float> *gradients);
};

class ReluChunked : public LayerChunked {
private:
    Relu relu_layer_;
    CudaHelper *cuda_helper_ = NULL;
    long chunk_size_;
    long last_chunk_size_;
    long num_chunks_;
    std::vector<Matrix<float>> *x_ = NULL;
    std::vector<Matrix<float>> y_;
    std::vector<Matrix<float>> gradients_;

public:
    ReluChunked();
    ReluChunked(CudaHelper *helper, long chunk_size, long num_nodes, long num_features);
    void set(CudaHelper *helper, long chunk_size, long num_nodes, long num_features);
    std::vector<Matrix<float>> *forward(std::vector<Matrix<float>> *x) override;
    std::vector<Matrix<float>> *backward(std::vector<Matrix<float>> *incoming_gradients) override;
};

class LogSoftmax : public Layer {
private:
    CudaHelper *cuda_helper_ = NULL;
    float alpha_;
    float beta_;
    Matrix<float> y_;
    Matrix<float> gradients_;

public:
    LogSoftmax();
    LogSoftmax(CudaHelper *helper);
    LogSoftmax(CudaHelper *helper, long num_nodes, long num_features);
    void set(CudaHelper *helper, long num_nodes, long num_features);
    Matrix<float> *forward(Matrix<float> *x);
    void forward(Matrix<float> *x, Matrix<float> *y);
    Matrix<float> *backward(Matrix<float> *incoming_gradients);
    void backward(Matrix<float> *incoming_gradients, Matrix<float> *y, Matrix<float> *gradients);
};

class LogSoftmaxChunked  : public LayerChunked {
private:
    LogSoftmax log_softmax_layer_;
    CudaHelper *cuda_helper_ = NULL;
    long chunk_size_;
    long last_chunk_size_;
    long num_chunks_;
    std::vector<Matrix<float>> y_;
    std::vector<Matrix<float>> gradients_;

public:
    LogSoftmaxChunked();
    LogSoftmaxChunked(CudaHelper *helper, long chunk_size, long num_nodes, long num_features);
    void set(CudaHelper *helper, long chunk_size, long num_nodes, long num_features);
    std::vector<Matrix<float>> *forward(std::vector<Matrix<float>> *x);
    std::vector<Matrix<float>> *backward(std::vector<Matrix<float>> *incoming_gradients);
};

#endif
