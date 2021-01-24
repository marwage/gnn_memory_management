// Copyright 2020 Marcel Wagenländer

#ifndef DROPOUT_H
#define DROPOUT_H

#include "cuda_helper.hpp"
#include "layer.hpp"
#include "tensors.hpp"

#include <vector>


class Dropout : public Layer {
protected:
    CudaHelper *cuda_helper_ = NULL;
    float probability_;
    unsigned long long seed_;
    size_t state_size_;
    char *reserve_space_ = NULL;
    size_t reserve_space_size_;
    Matrix<float> y_;
    Matrix<float> gradients_;

public:
    Dropout();
    Dropout(CudaHelper *helper, long num_nodes, long num_features);
    ~Dropout();
    void set(CudaHelper *helper, long num_nodes, long num_features);
    Matrix<float> *forward(Matrix<float> *x);
    void forward(Matrix<float> *x, Matrix<float> *y);
    Matrix<float> *backward(Matrix<float> *in_gradients);
    void backward(Matrix<float> *incoming_gradients, Matrix<float> *y, Matrix<float> *gradients);
};

class DropoutChunked : public LayerChunked {
protected:
    CudaHelper *cuda_helper_;
    float probability_;
    unsigned long long seed_;
    int chunk_size_;
    int last_chunk_size_;
    int num_chunks_;
    size_t state_size_;
    std::vector<char *> reserve_space_;
    size_t reserve_space_size_;
    std::vector<Matrix<float>> y_;
    std::vector<Matrix<float>> gradients_;

    bool keep_allocation_;
    void *d_states_;
    float *d_x_;
    float *d_y_;
    void *d_reserve_space_;
    cudnnDropoutDescriptor_t dropout_desc_;
    cudnnTensorDescriptor_t x_desc_;
    cudnnTensorDescriptor_t y_desc_;

    void allocate_gpu_memory();
    void free_gpu_memory();

public:
    DropoutChunked();
    DropoutChunked(CudaHelper *helper, int chunk_size, int num_nodes, long num_features);
    DropoutChunked(CudaHelper *helper, int chunk_size, int num_nodes, long num_features, bool keep_allocation);
    ~DropoutChunked();
    void set(CudaHelper *helper, long chunk_size, long num_nodes, long num_features) override;
    void set(CudaHelper *helper, long chunk_size, long num_nodes, long num_features, bool keep_allocation);
    std::vector<Matrix<float>> *forward(std::vector<Matrix<float>> *x) override;
    std::vector<Matrix<float>> *backward(std::vector<Matrix<float>> *incoming_gradients) override;
};

class DropoutPipelined : public LayerPipelined, public DropoutChunked {
protected:
    long num_steps_;
    std::vector<cudnnDropoutDescriptor_t> dropout_desc_;
    std::vector<cudnnTensorDescriptor_t> x_desc_;
    std::vector<cudnnTensorDescriptor_t> y_desc_;
    std::vector<cudnnTensorDescriptor_t> dx_desc_;
    std::vector<cudnnTensorDescriptor_t> dy_desc_;
    std::vector<float *> d_x_;
    std::vector<float *> d_y_;
    std::vector<float *> d_dx_;
    std::vector<float *> d_dy_;
    std::vector<char *> d_states_;
    std::vector<char *> d_reserve_space_;
    std::vector<Matrix<float>> *x_ = NULL;
    std::vector<Matrix<float>> *incoming_gradients_ = NULL;

public:
    DropoutPipelined();
    DropoutPipelined(CudaHelper *helper, long chunk_size, long num_nodes, long num_features);
    void set(CudaHelper *helper, long chunk_size, long num_nodes, long num_features) override;
    std::vector<Matrix<float>> *forward(std::vector<Matrix<float>> *x);
    std::vector<Matrix<float>> *backward(std::vector<Matrix<float>> *incoming_gradients) override;
    void forward_in(long chunk, long buffer) override;
    void forward_out(long chunk, long buffer) override;
    void forward_compute(long chunk, long buffer) override;
    void backward_in(long chunk, long buffer) override;
    void backward_out(long chunk, long buffer) override;
    void backward_compute(long chunk, long buffer) override;
};

#endif
