// Copyright 2020 Marcel Wagenländer

#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#include "adam.hpp"
#include "axpby.h"
#include "invsqrt.h"
#include "elesq.h"
#include "axdy.h"


Adam::Adam(CudaHelper *helper, float learning_rate, matrix<float> *parameters, int num_parameters) {
    cuda_helper_ = helper;
    learning_rate_ = learning_rate;
    num_parameters_ = num_parameters;
    t_ = 1;

    momentum_v_ = reinterpret_cast<matrix<float> *>(malloc(num_parameters * sizeof(matrix<float>)));
    momentum_m_ = reinterpret_cast<matrix<float> *>(malloc(num_parameters * sizeof(matrix<float>)));
    for (int i = 0; i < num_parameters; ++i) {
        momentum_v_[i].rows = parameters[i].rows;
        momentum_v_[i].columns = parameters[i].columns;
        momentum_v_[i].values = reinterpret_cast<float *>(
                malloc(momentum_v_[i].rows * momentum_v_[i].columns * sizeof(float)));
        for (int j = 0; j < momentum_v_[i].rows * momentum_v_[i].columns; ++j) {
            momentum_v_[i].values[j] = 0.0;
        }
        momentum_m_[i].rows = parameters[i].rows;
        momentum_m_[i].columns = parameters[i].columns;
        momentum_m_[i].values = reinterpret_cast<float *>(
                malloc(momentum_m_[i].rows * momentum_m_[i].columns * sizeof(float)));
        for (int j = 0; j < momentum_m_[i].rows * momentum_m_[i].columns; ++j) {
            momentum_m_[i].values[j] = 0.0;
        }
    }
}

matrix<float> *Adam::step(matrix<float> *gradients) {
    matrix<float> *update = reinterpret_cast<matrix<float> *>(
            malloc(num_parameters_ * sizeof(matrix<float>)));
    for (int i = 0; i < num_parameters_; ++i) {
        update[i].rows = gradients[i].rows;
        update[i].columns = gradients[i].columns;
        update[i].values = reinterpret_cast<float *>(
                malloc(update[i].rows * update[i].columns * sizeof(float)));
    }

    float *d_momentum_m;
    float *d_momentum_v;
    float *d_gradients;

    for (int i = 0; i < num_parameters_; ++i) {
        // momentum_m_[i] = beta_1_ * momentum_m_[i] + (1 - beta_1_) * gradients[i];
        check_cuda(cudaMalloc(&d_gradients, gradients[i].rows * gradients[i].columns * sizeof(float)));
        check_cuda(cudaMemcpy(d_gradients, gradients[i].values,
                              gradients[i].rows * gradients[i].columns * sizeof(float),
                              cudaMemcpyHostToDevice));

        check_cuda(cudaMalloc(&d_momentum_m, momentum_m_[i].rows * momentum_m_[i].columns * sizeof(float)));
        check_cuda(cudaMemcpy(d_momentum_m, momentum_m_[i].values,
                              momentum_m_[i].rows * momentum_m_[i].columns * sizeof(float),
                              cudaMemcpyHostToDevice));


        xpy((1 - beta_1_), d_gradients, beta_1_, d_momentum_m, momentum_m_[i].rows * momentum_m_[i].columns);

        // momentum_v_[i] = beta_2_ * momentum_v_[i] + (1 - beta_2_) * pow(gradients[i], 2);
        ele_squared(d_gradients, gradients[i].rows * gradients[i].columns);

        check_cuda(cudaMalloc(&d_momentum_v, momentum_v_[i].rows * momentum_v_[i].columns * sizeof(float)));
        check_cuda(cudaMemcpy(d_momentum_v, momentum_v_[i].values,
                              momentum_v_[i].rows * momentum_v_[i].columns * sizeof(float),
                              cudaMemcpyHostToDevice));

        xpy((1 - beta_2_), d_gradients, beta_2_, d_momentum_v, momentum_v_[i].rows * momentum_v_[i].columns);

        check_cuda(cudaMemcpy(momentum_m_[i].values, d_momentum_m,
                              momentum_m_[i].rows * momentum_m_[i].columns * sizeof(float),
                              cudaMemcpyDeviceToHost));
        check_cuda(cudaMemcpy(momentum_v_[i].values, d_momentum_v,
                              momentum_v_[i].rows * momentum_v_[i].columns * sizeof(float),
                              cudaMemcpyDeviceToHost));

        // learning_rate_t = learning_rate * sqrt(1 - beta_2 ^t) / (1 - beta_1 ^t)
        float learning_rate_t = learning_rate_ * sqrt(1 - pow(beta_2_, t_)) / (1 - pow(beta_1_, t_));

        // update[i] = learning_rate_t * momentum_m_ / (sqrt(momentum_v_) + epsilon_);
        inverse_sqrt(d_momentum_v, epsilon_, momentum_v_[i].rows * momentum_v_[i].columns);

        ax_dot_y(learning_rate_t, d_momentum_m, d_momentum_v, momentum_m_[i].rows * momentum_m_[i].columns);

        check_cuda(cudaMemcpy(update[i].values, d_momentum_v,
                              update[i].rows * update[i].columns * sizeof(float),
                              cudaMemcpyDeviceToHost));

        // clean-up
        check_cuda(cudaFree(d_momentum_m));
        check_cuda(cudaFree(d_gradients));
        check_cuda(cudaFree(d_momentum_v));
    }

    t_ = t_ + 1;

    return update;
}
