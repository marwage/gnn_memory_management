// Copyright 2020 Marcel Wagenländer

#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "cuda_helper.hpp"
#include "tensors.hpp"


class Relu {
private:
    float alpha_;
    float beta_;
    CudaHelper *cuda_helper_;
    cudnnActivationDescriptor_t relu_desc_;
    cudnnTensorDescriptor_t y_desc_;
    matrix<float> y_;
    matrix<float> x_;
    cudnnTensorDescriptor_t x_desc_;

public:
    Relu(CudaHelper *helper);

    matrix<float> forward(matrix<float> X);

    matrix<float> backward(matrix<float> in_gradients);
};

class LogSoftmax {
private:
    float alpha_;
    float beta_;
    matrix<float> y_;
    CudaHelper *cuda_helper_;

public:
    LogSoftmax(CudaHelper *helper);

    matrix<float> forward(matrix<float> X);

    matrix<float> backward(matrix<float> in_gradients);
};

#endif
