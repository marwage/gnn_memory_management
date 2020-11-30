// Copyright 2020 Marcel Wagenländer

#include <cmath>
#include <cuda_runtime.h>

#include "adam.hpp"
#include "axdy.h"
#include "axpby.h"
#include "elesq.h"
#include "invsqrt.h"


Adam::Adam(CudaHelper *helper, float learning_rate, Matrix<float> **parameters, int num_parameters) {
    cuda_helper_ = helper;
    learning_rate_ = learning_rate;
    num_parameters_ = num_parameters;
    t_ = 1;

    momentum_vs_ = std::vector<Matrix<float>>(num_parameters);
    momentum_ms_ = std::vector<Matrix<float>>(num_parameters);
    updates_ = std::vector<Matrix<float>>(num_parameters);
    for (int i = 0; i < num_parameters; ++i) {
        momentum_vs_[i].set(parameters[i]->num_rows_, parameters[i]->num_columns_, false);
        for (long j = 0; j < momentum_vs_[i].size_; ++j) {
            momentum_vs_[i].values_[j] = 0.0;
        }

        momentum_ms_[i].set(parameters[i]->num_rows_, parameters[i]->num_columns_, false);
        for (long j = 0; j < momentum_ms_[i].size_; ++j) {
            momentum_ms_[i].values_[j] = 0.0;
        }

        updates_[i].set(parameters[i]->num_rows_, parameters[i]->num_columns_, false);
    }
}

Matrix<float> *Adam::step(Matrix<float> **gradients) {
    for (int i = 0; i < num_parameters_; ++i) {
        to_column_major_inplace(gradients[i]);
    }

    float *d_momentum_m;
    float *d_momentum_v;
    float *d_gradients;

    for (int i = 0; i < num_parameters_; ++i) {
        // momentum_ms_[i] = beta_1_ * momentum_ms_[i] + (1 - beta_1_) * gradients[i];
        check_cuda(cudaMalloc(&d_gradients, gradients[i]->num_rows_ * gradients[i]->num_columns_ * sizeof(float)));
        check_cuda(cudaMemcpy(d_gradients, gradients[i]->values_,
                              gradients[i]->num_rows_ * gradients[i]->num_columns_ * sizeof(float),
                              cudaMemcpyHostToDevice));

        check_cuda(cudaMalloc(&d_momentum_m, momentum_ms_[i].num_rows_ * momentum_ms_[i].num_columns_ * sizeof(float)));
        check_cuda(cudaMemcpy(d_momentum_m, momentum_ms_[i].values_,
                              momentum_ms_[i].num_rows_ * momentum_ms_[i].num_columns_ * sizeof(float),
                              cudaMemcpyHostToDevice));


        xpy((1 - beta_1_), d_gradients, beta_1_, d_momentum_m, momentum_ms_[i].num_rows_ * momentum_ms_[i].num_columns_);

        // momentum_vs_[i] = beta_2_ * momentum_vs_[i] + (1 - beta_2_) * pow(gradients[i], 2);
        ele_squared(d_gradients, gradients[i]->num_rows_ * gradients[i]->num_columns_);

        check_cuda(cudaMalloc(&d_momentum_v, momentum_vs_[i].num_rows_ * momentum_vs_[i].num_columns_ * sizeof(float)));
        check_cuda(cudaMemcpy(d_momentum_v, momentum_vs_[i].values_,
                              momentum_vs_[i].num_rows_ * momentum_vs_[i].num_columns_ * sizeof(float),
                              cudaMemcpyHostToDevice));

        xpy((1 - beta_2_), d_gradients, beta_2_, d_momentum_v, momentum_vs_[i].num_rows_ * momentum_vs_[i].num_columns_);

        check_cuda(cudaMemcpy(momentum_ms_[i].values_, d_momentum_m,
                              momentum_ms_[i].num_rows_ * momentum_ms_[i].num_columns_ * sizeof(float),
                              cudaMemcpyDeviceToHost));
        check_cuda(cudaMemcpy(momentum_vs_[i].values_, d_momentum_v,
                              momentum_vs_[i].num_rows_ * momentum_vs_[i].num_columns_ * sizeof(float),
                              cudaMemcpyDeviceToHost));

        // learning_rate_t = learning_rate * sqrt(1 - beta_2 ^t) / (1 - beta_1 ^t)
        float learning_rate_t = learning_rate_ * sqrt(1 - pow(beta_2_, t_)) / (1 - pow(beta_1_, t_));

        // update[i] = learning_rate_t * momentum_ms_ / (sqrt(momentum_vs_) + epsilon_);
        inverse_sqrt(d_momentum_v, epsilon_, momentum_vs_[i].num_rows_ * momentum_vs_[i].num_columns_);

        ax_dot_y(learning_rate_t, d_momentum_m, d_momentum_v, momentum_ms_[i].num_rows_ * momentum_ms_[i].num_columns_);

        check_cuda(cudaMemcpy(updates_[i].values_, d_momentum_v,
                              updates_[i].num_rows_ * updates_[i].num_columns_ * sizeof(float),
                              cudaMemcpyDeviceToHost));

        // clean-up
        check_cuda(cudaFree(d_momentum_m));
        check_cuda(cudaFree(d_gradients));
        check_cuda(cudaFree(d_momentum_v));
    }

    t_ = t_ + 1;

    return updates_.data();// TODO now sure if it works with .data()
}
