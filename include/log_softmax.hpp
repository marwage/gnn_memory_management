// Copyright 2020 Marcel Wagenländer

#ifndef ALZHEIMER_LOG_SOFTMAX_H
#define ALZHEIMER_LOG_SOFTMAX_H

#include "cuda_helper.hpp"
#include "layer.hpp"
#include "tensors.hpp"

#include <vector>


class LogSoftmax : public Layer {
protected:
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

class LogSoftmaxChunked : public LayerChunked {
protected:
    CudaHelper *cuda_helper_ = NULL;
    float alpha_;
    float beta_;
    long chunk_size_;
    long last_chunk_size_;
    long num_chunks_;
    std::vector<Matrix<float>> y_;
    std::vector<Matrix<float>> gradients_;

    bool keep_allocation_;
    float *d_x_;
    cudnnTensorDescriptor_t x_desc_;
    float *d_y_;
    cudnnTensorDescriptor_t y_desc_;
    float *d_dy_;
    cudnnTensorDescriptor_t dy_desc_;

    void allocate_gpu_memory_forward();
    void free_gpu_memory_forward();
    void allocate_gpu_memory_backward();
    void free_gpu_memory_backward();
    void allocate_gpu_memory();
    void free_gpu_memory();

public:
    LogSoftmaxChunked();
    LogSoftmaxChunked(CudaHelper *helper, long chunk_size, long num_nodes, long num_features);
    LogSoftmaxChunked(CudaHelper *helper, long chunk_size, long num_nodes, long num_features, bool keep_allocation);
    ~LogSoftmaxChunked();
    void set(CudaHelper *helper, long chunk_size, long num_nodes, long num_features) override;
    void set(CudaHelper *helper, long chunk_size, long num_nodes, long num_features, bool keep_allocation) override;
    std::vector<Matrix<float>> *forward(std::vector<Matrix<float>> *x) override;
    std::vector<Matrix<float>> *backward(std::vector<Matrix<float>> *incoming_gradients) override;
};

class LogSoftmaxPipelined : public LayerPipelined, public LogSoftmaxChunked {
protected:
    long num_steps_;
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

public:
    LogSoftmaxPipelined();
    LogSoftmaxPipelined(CudaHelper *helper, long chunk_size, long num_nodes, long num_features);
    void set(CudaHelper *helper, long chunk_size, long num_nodes, long num_features);
    std::vector<Matrix<float>> *forward(std::vector<Matrix<float>> *x) override;
    std::vector<Matrix<float>> *backward(std::vector<Matrix<float>> *incoming_gradients) override;
    void forward_in(long chunk, long buffer) override;
    void forward_out(long chunk, long buffer) override;
    void forward_compute(long chunk, long buffer) override;
    void backward_in(long chunk, long buffer) override;
    void backward_out(long chunk, long buffer) override;
    void backward_compute(long chunk, long buffer) override;
};

#endif//ALZHEIMER_LOG_SOFTMAX_H
