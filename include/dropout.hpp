// Copyright 2020 Marcel Wagenländer

#ifndef DROPOUT_H
#define DROPOUT_H

#include "cuda_helper.hpp"
#include "layer.hpp"
#include "tensors.hpp"

#include <vector>


class Dropout : public Layer {
private:
    CudaHelper *cuda_helper_ = NULL;
    float probability_ = 0.2f;
    unsigned long long seed_;
    char *reserve_space_ = NULL;
    size_t reserve_space_size_;
    char *states_ = NULL;
    size_t state_size_;
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
private:
    CudaHelper *cuda_helper_ = NULL;
    int chunk_size_;
    int last_chunk_size_;
    int num_chunks_;
    std::vector<Dropout> dropout_layers_;
    std::vector<Matrix<float>> y_;
    std::vector<Matrix<float>> gradients_;

public:
    DropoutChunked();
    DropoutChunked(CudaHelper *helper, int chunk_size, int num_nodes, long num_features);
    void set(CudaHelper *helper, long chunk_size, long num_nodes, long num_features) override;
    std::vector<Matrix<float>> *forward(std::vector<Matrix<float>> *x) override;
    std::vector<Matrix<float>> *backward(std::vector<Matrix<float>> *incoming_gradients) override;
};

#endif
