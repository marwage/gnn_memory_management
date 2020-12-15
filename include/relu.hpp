// Copyright 2020 Marcel Wagenländer

#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "cuda_helper.hpp"
#include "layer.hpp"
#include "tensors.hpp"

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
    cudnnActivationDescriptor_t relu_desc_;
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

class ReluPipelined : public LayerChunked {
private:
    CudaHelper *cuda_helper_ = NULL;
    cudnnActivationDescriptor_t relu_desc_;
    long chunk_size_;
    long last_chunk_size_;
    long num_chunks_;
    float alpha_;
    float beta_;
    std::vector<cudnnTensorDescriptor_t> x_desc_;
    std::vector<cudnnTensorDescriptor_t> y_desc_;
    std::vector<cudnnTensorDescriptor_t> dx_desc_;
    std::vector<cudnnTensorDescriptor_t> dy_desc_;
    std::vector<float *> d_x_;
    std::vector<float *> d_y_;
    std::vector<float *> d_dx_;
    std::vector<float *> d_dy_;
    std::vector<Matrix<float>> *x_ = NULL;
    std::vector<Matrix<float>> *incoming_gradients_ = NULL;
    std::vector<Matrix<float>> y_;
    std::vector<Matrix<float>> gradients_;

public:
    ReluPipelined();
    ReluPipelined(CudaHelper *helper, long chunk_size, long num_nodes, long num_features);
    void set(CudaHelper *helper, long chunk_size, long num_nodes, long num_features);
    std::vector<Matrix<float>> *forward(std::vector<Matrix<float>> *x) override;
    std::vector<Matrix<float>> *backward(std::vector<Matrix<float>> *incoming_gradients) override;
    void forward_in(long chunk, long buffer);
    void forward_out(long chunk, long buffer);
    void forward_compute(long buffer);
    void backward_in(long chunk, long buffer);
    void backward_out(long chunk, long buffer);
    void backward_compute(long buffer);
    void pipeline(bool forward);
};

#endif
