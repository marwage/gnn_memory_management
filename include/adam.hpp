// Copyright 2020 Marcel Wagenländer

#ifndef ADAM_HPP
#define ADAM_HPP

#include "cuda_helper.hpp"
#include "tensors.hpp"

#include <vector>

class Adam {
private:
    float learning_rate_;
    long t_;
    const float beta_1_ = 0.9;
    const float beta_2_ = 0.999;
    const float epsilon_ = 1e-8;
    std::vector<Matrix<float> *> parameters_;
    std::vector<Matrix<float> *> gradients_;
    std::vector<Matrix<float>> momentum_ms_;
    std::vector<Matrix<float>> momentum_vs_;
    long num_parameters_;
    CudaHelper *cuda_helper_ = NULL;

public:
    Adam(CudaHelper *helper, float learning_rate, std::vector<Matrix<float> *> parameters, std::vector<Matrix<float> *> gradients);
    void step();
};

#endif//ADAM_HPP
