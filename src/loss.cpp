// Copyright 2020 Marcel Wagenländer

#include "loss.hpp"


NLLLoss::NLLLoss(long num_nodes, long num_features) {
    gradients_.set(num_nodes, num_features, false);
}

float NLLLoss::forward(Matrix<float> *x, Matrix<int> *labels) {
    to_column_major_inplace(x);

    float loss = 0.0;
    for (int i = 0; i < x->num_rows_; ++i) {
        loss = loss + x->values_[labels->values_[i] * x->num_rows_ + i];
    }
    loss = loss / (float) x->num_rows_;
    loss = -loss;

    input_ = x;
    labels_ = labels;

    return static_cast<float>(loss);
}

Matrix<float> *NLLLoss::backward() {
    gradients_.values_ = {};

    for (int i = 0; i < labels_->num_rows_; ++i) {
        gradients_.values_[labels_->values_[i] * labels_->num_rows_ + i] = -1.0 / input_->num_rows_;
    }

    return &gradients_;
}
