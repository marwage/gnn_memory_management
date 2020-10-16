// Copyright 2020 Marcel Wagenländer

#include <cmath>
#include <cuda_runtime.h>

#include "adam.hpp"
#include "axpby.h"
#include "invsqrt.h"


Adam::Adam(CudaHelper *helper, float learning_rate, matrix<float> *parameters, int num_parameters) {
    cuda_helper_ = helper;
    learning_rate_ = learning_rate;
    num_parameters_ = num_parameters;
    t_ = 0;

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
    float *d_gradients;
    float *d_gradients_square;
    float *d_momentum_v;
    float *d_momentum_m_hat;
    float *d_momentum_v_hat;

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
        check_cuda(cudaMalloc(&d_gradients_square, gradients[i].rows * gradients[i].columns * sizeof(float)));
        check_cuda(cudaMemcpy(d_gradients_square, gradients[i].values,
                              gradients[i].rows * gradients[i].columns * sizeof(float),
                              cudaMemcpyHostToDevice));

        float alpha = 1.0;
        check_cublas(cublasSaxpy(cuda_helper_->cublas_handle,
                                 gradients[i].rows * gradients[i].columns,
                                 &alpha, d_gradients, 1,
                                 d_gradients_square, 1));

        check_cuda(cudaMalloc(&d_momentum_v, momentum_v_[i].rows * momentum_v_[i].columns * sizeof(float)));
        check_cuda(cudaMemcpy(d_momentum_v, momentum_v_[i].values,
                              momentum_v_[i].rows * momentum_v_[i].columns * sizeof(float),
                              cudaMemcpyHostToDevice));

        xpy((1 - beta_2_), d_gradients_square, beta_2_, d_momentum_v, momentum_v_[i].rows * momentum_v_[i].columns);

        // momentum_m_hat = momentum_m_[i] / (1 - pow(beta_1_, t_));
        check_cuda(cudaMalloc(&d_momentum_m_hat, momentum_m_[i].rows * momentum_m_[i].columns * sizeof(float)));
        check_cuda(cudaMemset(d_momentum_m_hat, 0, momentum_m_[i].rows * momentum_m_[i].columns * sizeof(float)));

        alpha = 1 / (1 - pow(beta_1_, t_));
        check_cublas(cublasSaxpy(cuda_helper_->cublas_handle,
                                 momentum_m_[i].rows * momentum_m_[i].columns,
                                 &alpha, d_momentum_m, 1,
                                 d_momentum_m_hat, 1));

        // momentum_v_hat = momentum_v_[i] / (1 - pow(beta_2_, t_));
        check_cuda(cudaMalloc(&d_momentum_v_hat, momentum_v_[i].rows * momentum_v_[i].columns * sizeof(float)));
        check_cuda(cudaMemset(d_momentum_v_hat, 0, momentum_v_[i].rows * momentum_v_[i].columns * sizeof(float)));

        alpha = 1 / (1 - pow(beta_2_, t_));
        check_cublas(cublasSaxpy(cuda_helper_->cublas_handle,
                                 momentum_v_[i].rows * momentum_v_[i].columns,
                                 &alpha, d_momentum_v, 1,
                                 d_momentum_m_hat, 1));

        // update[i] = - learning_rate_ * momentum_m_hat / (sqrt(momentum_v_hat) + epsilon_);
        inverse_sqrt(d_momentum_v_hat, epsilon_, momentum_v_[i].rows * momentum_v_[i].columns);

        alpha = learning_rate_;
        check_cublas(cublasSaxpy(cuda_helper_->cublas_handle,
                                 momentum_m_[i].rows * momentum_m_[i].columns,
                                 &alpha, d_momentum_m_hat, 1,
                                 d_momentum_v_hat, 1));

        check_cuda(cudaMemcpy(update[i].values, d_momentum_v_hat,
                              update[i].rows * update[i].columns * sizeof(float),
                              cudaMemcpyDeviceToHost));

        // clean-up
        check_cuda(cudaFree(d_momentum_m));
        check_cuda(cudaFree(d_gradients));
        check_cuda(cudaFree(d_gradients_square));
        check_cuda(cudaFree(d_momentum_v));
        check_cuda(cudaFree(d_momentum_m_hat));
        check_cuda(cudaFree(d_momentum_v_hat));
    }

    t_ = t_ + 1;

    return update;
}
