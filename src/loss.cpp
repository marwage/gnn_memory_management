// Copyright 2020 Marcel Wagenländer

#include "loss.hpp"


NLLLoss::NLLLoss(long num_nodes, long num_features) {
    num_nodes_ = num_nodes;
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

    labels_ = labels;

    return static_cast<float>(loss);
}

float NLLLoss::forward(std::vector<Matrix<float>> *x, Matrix<int> *labels) {
    long num_chunks = x->size();
    for (int i = 0; i < num_chunks; ++i) {
        to_row_major_inplace(&x->at(i));
    }

    double loss = 0.0;
    long row = 0;
    long chunk_size = x->at(0).num_rows_;
    for (int i = 0; i < num_chunks; ++i) {
        for (int j = 0; j < x->at(i).num_rows_; ++j) {
            row = i * chunk_size + j;
            loss = loss + x->at(i).values_[j * x->at(i).num_columns_ + labels->values_[row]];
        }
    }

    loss = loss / (double) num_nodes_;
    loss = -loss;

    labels_ = labels;

    return static_cast<float>(loss);
}

Matrix<float> *NLLLoss::backward() {
    gradients_.set_values(0.0);

    for (int i = 0; i < labels_->num_rows_; ++i) {
        gradients_.values_[labels_->values_[i] * labels_->num_rows_ + i] = -1.0 / num_nodes_;
    }

    return &gradients_;
}
