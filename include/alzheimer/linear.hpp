// Copyright 2020 Marcel Wagenländer

#ifndef LINEAR_H
#define LINEAR_H

#include "tensors.hpp"

class Linear {
 private:
    int num_in_features, num_out_features;
    matrix<float> weight;  // column-major
    vector<float> bias;

    void init_weight_bias();
    matrix<float> expand_bias(int num_rows);
 public:
    Linear();
    Linear(int in_features, int out_features);
    matrix<float>* get_parameters();
    matrix<float> forward(matrix<float> X);
};

#endif

