// Copyright 2020 Marcel Wagenländer

#ifndef LOSS_H
#define LOSS_H

#include "tensors.hpp"


class NLLLoss {
private:
    matrix<float> *input_;
    matrix<int> *labels_;
    matrix<float> gradients_;

public:
    NLLLoss(long num_nodes, long num_features);
    float forward(matrix<float> *x, matrix<int> *y);
    matrix<float>* backward();
};

#endif
