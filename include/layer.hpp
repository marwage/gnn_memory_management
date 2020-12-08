// Copyright 2020 Marcel Wagenländer

#ifndef ALZHEIMER_LAYER_H
#define ALZHEIMER_LAYER_H

#include "tensors.hpp"

#include <vector>


class Layer {
public:
    virtual Matrix<float> *forward(Matrix<float> *x) = 0;
    virtual Matrix<float> *backward(Matrix<float> *incoming_gradients) = 0;
    virtual void set(CudaHelper *helper, long num_nodes, long num_features) = 0;
};

class LayerChunked {
public:
    virtual std::vector<Matrix<float>> *forward(std::vector<Matrix<float>> *x) = 0;
    virtual std::vector<Matrix<float>> *backward(std::vector<Matrix<float>> *incoming_gradients) = 0;
    virtual void set(CudaHelper *helper, long chunk_size, long num_nodes, long num_features) = 0;
};

#endif//ALZHEIMER_LAYER_H