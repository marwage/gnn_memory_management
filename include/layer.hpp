// Copyright 2020 Marcel Wagenländer

#ifndef ALZHEIMER_LAYER_H
#define ALZHEIMER_LAYER_H

#include "tensors.hpp"

#include <vector>


class Layer {
protected:
    CudaHelper *cuda_helper_ = NULL;
    Matrix<float> y_;
    Matrix<float> gradients_;

public:
    virtual Matrix<float> *forward(Matrix<float> *x) = 0;
    virtual Matrix<float> *backward(Matrix<float> *incoming_gradients) = 0;
    virtual void set(CudaHelper *helper, long num_nodes, long num_features) = 0;
};

class LayerChunked {
protected:
    long chunk_size_;
    long last_chunk_size_;
    long num_chunks_;
    CudaHelper *cuda_helper_ = NULL;
    std::vector<Matrix<float>> y_;
    std::vector<Matrix<float>> gradients_;

public:
    virtual std::vector<Matrix<float>> *forward(std::vector<Matrix<float>> *x) = 0;
    virtual std::vector<Matrix<float>> *backward(std::vector<Matrix<float>> *incoming_gradients) = 0;
    virtual void set(CudaHelper *helper, long chunk_size, long num_nodes, long num_features) = 0;
};

class LayerPipelined {
public:
    virtual void forward_in(long chunk, long buffer) = 0;
    virtual void forward_out(long chunk, long buffer) = 0;
    virtual void forward_compute(long buffer) = 0;
    virtual void backward_in(long chunk, long buffer) = 0;
    virtual void backward_out(long chunk, long buffer) = 0;
    virtual void backward_compute(long buffer) = 0;
    void pipeline(bool forward, long num_chunks);
};

#endif//ALZHEIMER_LAYER_H