// Copyright 2020 Marcel Wagenländer

#ifndef LINEAR_H
#define LINEAR_H

#include "cuda_helper.hpp"
#include "tensors.hpp"

class Linear {
private:
    int num_in_features_, num_out_features_;
    matrix<float> weight_;
    matrix<float> bias_;
    matrix<float> grad_weight_;
    matrix<float> grad_bias_;
    matrix<float> x_;
    CudaHelper *cuda_helper_;

    void init_weight_bias();
    matrix<float> expand_bias(int num_rows);

public:
    Linear();
    Linear(int in_features, int out_features, CudaHelper *helper);
    matrix<float> *get_parameters();
    matrix<float> *get_gradients();
    matrix<float> forward(matrix<float> X);
    matrix<float> backward(matrix<float> in_gradients);
    void update_weights(matrix<float> *gradients);
};

#endif
