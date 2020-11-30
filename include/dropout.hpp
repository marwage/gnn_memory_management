// Copyright 2020 Marcel Wagenländer

#ifndef DROPOUT_H
#define DROPOUT_H

#include "cuda_helper.hpp"
#include "tensors.hpp"
#include <vector>


class DropoutParent {
public:
    virtual Matrix<float> *forward(Matrix<float> *x) = 0;
    virtual Matrix<float> *backward(Matrix<float> *in_gradients) = 0;
};

class Dropout : public DropoutParent {
private:
    CudaHelper *cuda_helper_ = NULL;
    cudnnDropoutDescriptor_t dropout_desc_;
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
    Matrix<float> *forward(Matrix<float> *x);
    Matrix<float> *backward(Matrix<float> *in_gradients);
};

class DropoutChunked : public DropoutParent {
private:
    CudaHelper *cuda_helper_ = NULL;
    int chunk_size_;
    int last_chunk_size_;
    int num_chunks_;
    std::vector<Dropout> dropout_layers_;
    Matrix<float> y_;
    Matrix<float> gradients_;

public:
    DropoutChunked(CudaHelper *helper, int chunk_size, int num_nodes, long num_features);
    Matrix<float> *forward(Matrix<float> *x);
    Matrix<float> *backward(Matrix<float> *in_gradients);
};

#endif
