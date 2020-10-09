// Copyright 2020 Marcel Wagenländer

#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "tensors.hpp"
#include "cuda_helper.hpp"


class Relu {
private:
    CudaHelper *cuda_helper_;
public:
    Relu(CudaHelper *helper);
    matrix<float> forward(matrix<float> X);
};

class LogSoftmax {
private:
    CudaHelper *cuda_helper_;
public:
    LogSoftmax(CudaHelper * helper);
    matrix<float> forward(matrix<float> X);
};

#endif
