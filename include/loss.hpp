// Copyright 2020 Marcel Wagenländer

#ifndef LOSS_H
#define LOSS_H

#include "tensors.hpp"


class NLLLoss {
private:
    long num_nodes_;
    Matrix<int> *labels_ = NULL;
    Matrix<float> gradients_;

public:
    NLLLoss(long num_nodes, long num_features);
    float forward(Matrix<float> *x, Matrix<int> *labels);
    float forward(std::vector<Matrix<float>> *x, Matrix<int> *labels);
    Matrix<float> *backward();
};

#endif
