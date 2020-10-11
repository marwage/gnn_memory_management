// Copyright 2020 Marcel Wagenländer

#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "tensors.hpp"
#include "cuda_helper.hpp"


class Relu {
private:
    float alpha_;
    float beta_;
    CudaHelper *cuda_helper_;
public:
    Relu(CudaHelper *helper);

    matrix<float> forward(matrix<float> X);
};

class LogSoftmax {
private:
    float alpha_;
    float beta_;
    matrix<float> Y_;
    CudaHelper *cuda_helper_;
public:
    LogSoftmax(CudaHelper *helper);

    matrix<float> forward(matrix<float> X);

    matrix<float> backward(matrix<float> in_gradients);
};

#endif
