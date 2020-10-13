// Copyright 2020 Marcel Wagenländer

#include "loss.hpp"

// debug
#include <iostream>


NLLLoss::NLLLoss() {}

float NLLLoss::forward(matrix<float> X, matrix<int> y) {
    float loss = 0.0;
    for (int i = 0; i < X.rows; ++i) {
        loss = loss + X.values[i * X.columns + y.values[i]];
    }
    loss = loss / (float) X.rows;
    loss = - loss;

    input_ = X;
    y_ = y;

    return static_cast<float>(loss);
}

matrix<float> NLLLoss::backward() {
    matrix<float> gradients;
    gradients.rows = input_.rows;
    gradients.columns = input_.columns;
    gradients.values = reinterpret_cast<float *>(
            malloc(gradients.rows * gradients.columns * sizeof(float)));
    for (int i = 0; i < gradients.rows * gradients.columns; ++i) {
        gradients.values[i] = 0.0;
    }

    for (int i = 0; i < y_.rows; ++i) {
        gradients.values[y_.values[i] * y_.rows + i] = - 1.0 / input_.rows;
    }

    return gradients;
}
