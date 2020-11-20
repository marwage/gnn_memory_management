// Copyright 2020 Marcel Wagenländer

#ifndef ADAM_HPP
#define ADAM_HPP

#include "cuda_helper.hpp"
#include "tensors.hpp"

class Adam {
private:
    float learning_rate_;
    int t_;
    const float beta_1_ = 0.9;
    const float beta_2_ = 0.999;
    const float epsilon_ = 1e-8;
    matrix<float> *momentum_ms_;
    matrix<float> *momentum_vs_;
    int num_parameters_;
    CudaHelper *cuda_helper_;
    matrix<float> *updates_;

public:
    Adam(CudaHelper *helper, float learning_rate, matrix<float> *parameters, int num_parameters);
    matrix<float> *step(matrix<float> *gradients);
};

#endif//ADAM_HPP
