// Copyright 2020 Marcel Wagenländer

#ifndef DROPOUT_H
#define DROPOUT_H

#include "tensors.hpp"
#include "cuda_helper.hpp"


class Dropout {
private:
    CudaHelper *cuda_helper_;
public:
    Dropout(CudaHelper *helper);
    matrix<float> forward(matrix<float> X);

};

#endif
