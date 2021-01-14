// Copyright 2020 Marcel Wagenländer

#include <cmath>
#include <cuda_runtime.h>

#include "adam.hpp"
#include "axdy.h"
#include "axpby.h"
#include "elesq.h"
#include "invsqrt.h"


Adam::Adam(CudaHelper *helper, float learning_rate, std::vector<Matrix<float> *> parameters, std::vector<Matrix<float> *> gradients) {
    if (parameters.size() != gradients.size()) {
        throw "Number of parameters and gradients is unequal";
    }
    cuda_helper_ = helper;
    learning_rate_ = learning_rate;
    num_parameters_ = parameters.size();
    t_ = 1;
    parameters_ = parameters;
    gradients_ = gradients;

    momentum_vs_ = std::vector<Matrix<float>>(num_parameters_);
    momentum_ms_ = std::vector<Matrix<float>>(num_parameters_);
    for (int i = 0; i < num_parameters_; ++i) {
        momentum_vs_[i].set(parameters[i]->num_rows_, parameters[i]->num_columns_, false);
        momentum_vs_[i].set_values(0.0);

        momentum_ms_[i].set(parameters[i]->num_rows_, parameters[i]->num_columns_, false);
        momentum_ms_[i].set_values(0.0);
    }
}

void Adam::step() {
    for (int i = 0; i < num_parameters_; ++i) {
        to_column_major_inplace(gradients_[i]);
    }

    long max_size = 0;
    for (long i = 0; i < num_parameters_; ++i) {
        if (parameters_.at(i)->size_ > max_size) {
            max_size = parameters_.at(i)->size_;
        }
    }

    float *d_gradients;
    check_cuda(cudaMalloc(&d_gradients, max_size * sizeof(float)));
    float *d_momentum_m;
    check_cuda(cudaMalloc(&d_momentum_m, max_size * sizeof(float)));
    float *d_momentum_v;
    check_cuda(cudaMalloc(&d_momentum_v, max_size * sizeof(float)));
    float *d_parameter;
    check_cuda(cudaMalloc(&d_parameter, max_size * sizeof(float)));

    for (int i = 0; i < num_parameters_; ++i) {
        // momentum_ms_[i] = beta_1_ * momentum_ms_[i] + (1 - beta_1_) * gradients[i];
        check_cuda(cudaMemcpy(d_gradients, gradients_[i]->values_,
                              gradients_[i]->num_rows_ * gradients_[i]->num_columns_ * sizeof(float),
                              cudaMemcpyHostToDevice));

        check_cuda(cudaMemcpy(d_momentum_m, momentum_ms_[i].values_,
                              momentum_ms_[i].num_rows_ * momentum_ms_[i].num_columns_ * sizeof(float),
                              cudaMemcpyHostToDevice));

        xpy((1 - beta_1_), d_gradients, beta_1_, d_momentum_m, momentum_ms_[i].num_rows_ * momentum_ms_[i].num_columns_);

        // momentum_vs_[i] = beta_2_ * momentum_vs_[i] + (1 - beta_2_) * pow(gradients[i], 2);
        ele_squared(d_gradients, gradients_[i]->num_rows_ * gradients_[i]->num_columns_);

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

        check_cuda(cudaMemcpy(d_parameter, parameters_[i]->values_,
                              parameters_[i]->size_ * sizeof(float),
                              cudaMemcpyHostToDevice));

        float alpha = -1.0;
        check_cublas(cublasSaxpy(cuda_helper_->cublas_handle,
                                 parameters_[i]->size_,
                                 &alpha, d_momentum_v, 1,
                                 d_parameter, 1));

        check_cuda(cudaMemcpy(parameters_[i]->values_, d_parameter,
                              parameters_[i]->size_ * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }

    check_cuda(cudaFree(d_parameter));
    check_cuda(cudaFree(d_momentum_m));
    check_cuda(cudaFree(d_gradients));
    check_cuda(cudaFree(d_momentum_v));

    t_ = t_ + 1;
}
