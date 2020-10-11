// Copyright 2020 Marcel Wagenländer

#ifndef SAGE_LINEAR_H
#define SAGE_LINEAR_H

#include "tensors.hpp"
#include "linear.hpp"
#include "cuda_helper.hpp"


class SageLinear {
 private:
    int num_in_features_;
    int num_out_features_;
    Linear linear_self_;
    Linear linear_neigh_;

    CudaHelper *cuda_helper_;
 public:
    struct SageLinearGradients {
        matrix<float> self_grads;
        matrix<float> neigh_grads;
    };
    SageLinear(int in_features, int out_features, CudaHelper *helper);
    matrix<float>* get_parameters();
    matrix<float> forward(matrix<float> features, matrix<float> aggr);
    SageLinearGradients backward(matrix<float> in_gradients);
    void update_weights(float learning_rate);
};

#endif
