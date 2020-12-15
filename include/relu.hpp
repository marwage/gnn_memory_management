// Copyright 2020 Marcel Wagenländer

#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "cuda_helper.hpp"
#include "layer.hpp"
#include "tensors.hpp"

#include <vector>


class Relu : public Layer {
protected:
    float alpha_;
    float beta_;
    cudnnActivationDescriptor_t relu_desc_;
    Matrix<float> *x_ = NULL;

public:
    Relu();
    Relu(CudaHelper *helper, long num_nodes, long num_features);
    void set(CudaHelper *helper, long num_nodes, long num_features) override;
    Matrix<float> *forward(Matrix<float> *x) override;
    Matrix<float> *backward(Matrix<float> *incoming_gradients) override;
};

class ReluChunked : public LayerChunked {
protected:
    float alpha_;
    float beta_;
    cudnnActivationDescriptor_t relu_desc_;
    std::vector<Matrix<float>> *x_ = NULL;

public:
    ReluChunked();
    ReluChunked(CudaHelper *helper, long chunk_size, long num_nodes, long num_features);
    void set(CudaHelper *helper, long chunk_size, long num_nodes, long num_features);
    std::vector<Matrix<float>> *forward(std::vector<Matrix<float>> *x) override;
    std::vector<Matrix<float>> *backward(std::vector<Matrix<float>> *incoming_gradients) override;
};

class ReluPipelined : public LayerPipelined, ReluChunked {
private:
    std::vector<cudnnTensorDescriptor_t> x_desc_;
    std::vector<cudnnTensorDescriptor_t> y_desc_;
    std::vector<cudnnTensorDescriptor_t> dx_desc_;
    std::vector<cudnnTensorDescriptor_t> dy_desc_;
    std::vector<float *> d_x_;
    std::vector<float *> d_y_;
    std::vector<float *> d_dx_;
    std::vector<float *> d_dy_;
    std::vector<Matrix<float>> *incoming_gradients_ = NULL;

public:
    ReluPipelined();
    ReluPipelined(CudaHelper *helper, long chunk_size, long num_nodes, long num_features);
    void set(CudaHelper *helper, long chunk_size, long num_nodes, long num_features);
    std::vector<Matrix<float>> *forward(std::vector<Matrix<float>> *x) override;
    std::vector<Matrix<float>> *backward(std::vector<Matrix<float>> *incoming_gradients) override;
    void forward_in(long chunk, long buffer) override;
    void forward_out(long chunk, long buffer) override;
    void forward_compute(long buffer) override;
    void backward_in(long chunk, long buffer) override;
    void backward_out(long chunk, long buffer) override;
    void backward_compute(long buffer) override;
};

#endif
