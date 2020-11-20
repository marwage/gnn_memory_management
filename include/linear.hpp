// Copyright 2020 Marcel Wagenländer

#ifndef LINEAR_H
#define LINEAR_H

#include "cuda_helper.hpp"
#include "tensors.hpp"

class Linear {
private:
    CudaHelper *cuda_helper_;
    long num_in_features_;
    long num_out_features_;
    matrix<float> weight_;
    matrix<float> bias_;
    matrix<float> bias_expanded_;
    matrix<float> grad_weight_;
    matrix<float> grad_bias_;
    matrix<float> x_;
    matrix<float> y_;
    float *ones_;
    matrix<float> gradients_input_;

    void init_weight_bias();
    matrix<float> expand_bias();

public:
    Linear();
    Linear(CudaHelper *helper, long in_features, long out_features, long num_nodes);
    matrix<float> *get_parameters();
    void set_parameters(matrix<float> *parameters);
    matrix<float> *get_gradients();
    void set_gradients(matrix<float> *grads);
    matrix<float> forward(matrix<float> X);
    matrix<float> backward(matrix<float> in_gradients);
    void update_weights(matrix<float> *gradients);
};

#endif
