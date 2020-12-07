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
    std::vector<Matrix<float> *> get_parameters();
    std::vector<Matrix<float> *> get_gradients();
    void set_gradients(Matrix<float> *weight_grads, Matrix<float> *bias_grads);
    Matrix<float> *forward(Matrix<float> *x);
    Matrix<float> *backward(Matrix<float> *in_gradients);
    void backward(Matrix<float> *incoming_gradients, Matrix<float> *x, Matrix<float> *gradients);
};

#endif
