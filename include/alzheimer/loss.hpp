// Copyright 2020 Marcel Wagenländer

#ifndef LOSS_H
#define LOSS_H

#include "tensors.hpp"


class NLLLoss {
private:
    matrix<float> input_;
    matrix<int> y_;

public:
    NLLLoss();
    float forward(matrix<float> X, matrix<int> y);
    matrix<float> backward();
};

#endif
