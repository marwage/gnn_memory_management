// Copyright 2020 Marcel Wagenländer

#ifndef DROPOUT_H
#define DROPOUT_H

#include "tensors.hpp"
#include "cuda_helper.hpp"


class Dropout {
private:
    CudaHelper *cuda_helper_;
    void *reserve_space_;
    size_t reserve_space_size_;
    cudnnDropoutDescriptor_t dropout_desc_;
    void * states_;
    size_t state_size_;
public:
    Dropout(CudaHelper *helper);
    matrix<float> forward(matrix<float> X);
    matrix<float> backward(matrix<float> in_gradients);
};

#endif
