// Copyright 2020 Marcel Wagenländer

#ifndef LINEAR_H
#define LINEAR_H

#include "tensors.hpp"
#include "cuda_helper.hpp"

class Linear {
 private:
    int num_in_features_, num_out_features_;
    matrix<float> weight_;
    matrix<float> bias_;
    matrix<float> grad_weight_;
    matrix<float> grad_bias_;
    CudaHelper *cuda_helper_;

    void init_weight_bias();
    matrix<float> expand_bias(int num_rows);
 public:
    Linear();
    Linear(int in_features, int out_features, CudaHelper *helper);
    matrix<float>* get_parameters();
    matrix<float> forward(matrix<float> X);
    matrix<float> backward(matrix<float> in_gradients);
    void update_weights(float learning_rate);
};

#endif

