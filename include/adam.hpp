// Copyright 2020 Marcel Wagenländer

#ifndef ADAM_HPP
#define ADAM_HPP

#include "cuda_helper.hpp"
#include "tensors.hpp"

#include <vector>

class Adam {
private:
    float learning_rate_;
    int t_;
    const float beta_1_ = 0.9;
    const float beta_2_ = 0.999;
    const float epsilon_ = 1e-8;
    std::vector<Matrix<float>> momentum_ms_;
    std::vector<Matrix<float>> momentum_vs_;
    int num_parameters_;
    CudaHelper *cuda_helper_ = NULL;
    std::vector<Matrix<float>> updates_;

public:
    Adam(CudaHelper *helper, float learning_rate, Matrix<float> **parameters, int num_parameters);
    Matrix<float> *step(Matrix<float> **gradients);
};

#endif//ADAM_HPP
