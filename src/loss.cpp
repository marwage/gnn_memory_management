// Copyright 2020 Marcel Wagenländer

#include "loss.hpp"

// debug
#include <iostream>


NLLLoss::NLLLoss(long num_nodes, long num_features) {
    gradients_ = new_float_matrix(num_nodes, num_features, false);
}

float NLLLoss::forward(matrix<float> *x, matrix<int> *labels) {
    to_column_major_inplace(x);

    float loss = 0.0;
    for (int i = 0; i < x->rows; ++i) {
        loss = loss + x->values[labels->values[i] * x->rows + i];
    }
    loss = loss / (float) x->rows;
    loss = -loss;

    input_ = x;
    labels_ = labels;

    return static_cast<float>(loss);
}

matrix<float>* NLLLoss::backward() {
    for (int i = 0; i < gradients_.rows * gradients_.columns; ++i) {
        gradients_.values[i] = 0.0;
    }

    for (int i = 0; i < labels_->rows; ++i) {
        gradients_.values[labels_->values[i] * labels_->rows + i] = -1.0 / input_->rows;
    }

    return &gradients_;
}
