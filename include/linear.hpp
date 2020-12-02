// Copyright 2020 Marcel Wagenländer

#ifndef LINEAR_H
#define LINEAR_H

#include "cuda_helper.hpp"
#include "tensors.hpp"

#include <vector>

class Linear {
private:
    CudaHelper *cuda_helper_;
    long num_in_features_;
    long num_out_features_;
    Matrix<float> weight_;
    Matrix<float> bias_;
    Matrix<float> bias_expanded_;
    Matrix<float> grad_weight_;
    Matrix<float> grad_bias_;
    Matrix<float> y_;
    std::vector<float> ones_;
    Matrix<float> gradients_input_;
    Matrix<float> *x_ = NULL;

    void init_weight_bias();
    void expand_bias();

public:
    Linear();
    Linear(CudaHelper *helper, long in_features, long out_features, long num_nodes);
    void set(CudaHelper *helper, long in_features, long out_features, long num_nodes);
    Matrix<float> **get_parameters();
    void set_parameters(Matrix<float> **parameters);
    void set_parameters(Matrix<float> *weight, Matrix<float> *bias);
    Matrix<float> **get_gradients();
    void set_gradients(Matrix<float> **grads);
    Matrix<float> *forward(Matrix<float> *x);
    Matrix<float> *backward(Matrix<float> *in_gradients);
    Matrix<float> *backward(Matrix<float> *in_gradients, Matrix<float> *x);
    void update_weights(Matrix<float> *gradients);
};

#endif
