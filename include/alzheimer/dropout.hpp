// Copyright 2020 Marcel Wagenländer

#ifndef DROPOUT_H
#define DROPOUT_H

#include "cuda_helper.hpp"
#include "tensors.hpp"
#include <vector>

class Dropout {
private:
    CudaHelper *cuda_helper_;
    void *reserve_space_;
    size_t reserve_space_size_;
    cudnnDropoutDescriptor_t dropout_desc_;
    void *states_;
    size_t state_size_;
    // std::vector <void *> reserve_spaces_;
    // int num_chunks_;

public:
    Dropout(CudaHelper *helper);
    matrix<float> forward(matrix<float> X);
    // matrix<float> forward_chunked(matrix<float> X, int chunk_size);
    matrix<float> backward(matrix<float> in_gradients);
    // matrix<float> backward_chunked(matrix<float> in_gradients, int chunk_size);
};

class DropoutChunked {
private:
    CudaHelper *cuda_helper_;
    int chunk_size_;
    int last_chunk_size_;
    int num_chunks_;
    Dropout *dropout_layers_;

public:
    DropoutChunked(CudaHelper *helper, int chunk_size);
    matrix<float> forward(matrix<float> X);
    matrix<float> backward(matrix<float> in_gradients);
};

#endif
