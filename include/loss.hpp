// Copyright 2020 Marcel Wagenländer

#ifndef LOSS_H
#define LOSS_H

#include "tensors.hpp"


class NLLLoss {
private:
    Matrix<float> *input_;
    Matrix<int> *labels_;
    Matrix<float> gradients_;

public:
    NLLLoss(long num_nodes, long num_features);
    float forward(Matrix<float> *x, Matrix<int> *y);
    Matrix<float>* backward();
};

#endif
