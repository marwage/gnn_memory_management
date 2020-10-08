// Copyright 2020 Marcel Wagenländer

#include "loss.hpp"


float negative_log_likelihood_loss(matrix<float> X, vector<int> y) {
    double loss = 0.0;
    for (int i = 0; i < X.rows; ++i) {
        loss = loss + X.values[i * X.columns + y.values[i]];
    }
    loss = loss / X.columns;
    loss = - loss;
    return static_cast<float>(loss);
}
