// Copyright 2020 Marcel Wagenländer

#include <chrono>
#include <cuda_runtime.h>
#include <random>

#include "cuda_helper.hpp"
#include "linear.hpp"
#include "tensors.hpp"


Linear::Linear() {}

Linear::Linear(CudaHelper *helper, long in_features, long out_features, long num_nodes) {
    set(helper, in_features, out_features, num_nodes);
}

void Linear::set(CudaHelper *helper, long in_features, long out_features, long num_nodes) {
    cuda_helper_ = helper;

    num_nodes_ = num_nodes;
    num_in_features_ = in_features;
    num_out_features_ = out_features;

    weight_.set(num_in_features_, num_out_features_, false);
    bias_.set(num_out_features_, 1, false);

    grad_weight_.set(weight_.num_rows_, weight_.num_columns_, false);
    grad_bias_.set(bias_.num_rows_, bias_.num_columns_, false);

    init_weight_bias();

    bias_expanded_.set(num_nodes, bias_.num_rows_, false);

    y_.set(num_nodes, weight_.num_columns_, false);

    ones_ = std::vector<float>(num_nodes, 1.0);

    gradients_input_.set(num_nodes, in_features, false);
}

void Linear::init_weight_bias() {
    double k = 1.0 / static_cast<double>(num_in_features_);
    k = sqrt(k);
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_real_distribution<float> distr(-k, k);
    for (int i = 0; i < weight_.num_rows_ * weight_.num_columns_; ++i) {
        weight_.values_[i] = distr(generator);
    }
    for (int i = 0; i < bias_.num_rows_ * bias_.num_columns_; ++i) {
        bias_.values_[i] = distr(generator);
    }
}

std::vector<Matrix<float> *> Linear::get_parameters() {
    std::vector<Matrix<float> *> parameters(2);
    parameters[0] = &weight_;
    parameters[1] = &bias_;

    return parameters;
}

std::vector<Matrix<float> *> Linear::get_gradients() {
    std::vector<Matrix<float> *> gradients(2);
    gradients[0] = &grad_weight_;
    gradients[1] = &grad_bias_;

    return gradients;
}

std::vector<float *> Linear::get_gradients_cuda() {
    std::vector<float *> gradients(2);
    gradients[0] = d_dweight_;
    gradients[1] = d_db_;

    return gradients;
}

void Linear::set_gradients(Matrix<float> *weight_grads, Matrix<float> *bias_grads) {
    to_column_major_inplace(weight_grads);
    to_column_major_inplace(bias_grads);

    std::memcpy(grad_weight_.values_, weight_grads->values_, grad_weight_.size_ * sizeof(float));
    std::memcpy(grad_bias_.values_, bias_grads->values_, grad_bias_.size_ * sizeof(float));
}

void Linear::expand_bias() {
    for (int i = 0; i < bias_expanded_.num_columns_; ++i) {
        for (int j = 0; j < bias_expanded_.num_rows_; ++j) {
            bias_expanded_.values_[i * bias_expanded_.num_rows_ + j] = bias_.values_[i];
        }
    }
}

void Linear::forward_init() {
    to_column_major_inplace(&weight_);
    to_column_major_inplace(&bias_);

    check_cuda(cudaMalloc(&d_weight_, weight_.size_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_weight_, weight_.values_, weight_.size_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    expand_bias();
    check_cuda(cudaMalloc(&d_bias_, bias_expanded_.size_ * sizeof(float)));
}

float *Linear::forward_compute(float *d_x, long num_rows) {
    // needs to be reset at every call because it's overwritten with the result
    check_cuda(cudaMemcpy(d_bias_, bias_expanded_.values_, bias_expanded_.size_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    float alpha = 1.0;
    float beta = 1.0;
    check_cublas(cublasSgemm(cuda_helper_->cublas_handle, // PyTorch uses GEMM too
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             num_rows, num_out_features_, num_in_features_,
                             &alpha,
                             d_x, num_rows,
                             d_weight_, weight_.num_rows_,
                             &beta,
                             d_bias_, num_rows));

    return d_bias_;
}

void Linear::forward_free() {
    check_cuda(cudaFree(d_weight_));
    check_cuda(cudaFree(d_bias_));
}

Matrix<float> *Linear::forward(Matrix<float> *x) {
    to_column_major_inplace(x);
    x_ = x;

    forward_init();

    float *d_x;
    check_cuda(cudaMalloc(&d_x, x->size_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_x, x->values_, x->size_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    forward_compute(d_x, num_nodes_);

    // get result of linear
    check_cuda(cudaMemcpy(y_.values_, d_bias_, y_.size_ * sizeof(float),
                          cudaMemcpyDeviceToHost));
    y_.is_row_major_ = false;

    // free
    check_cuda(cudaFree(d_x));
    forward_free();

    return &y_;
}

void Linear::backward_init() {
    check_cuda(cudaMalloc(&d_ones_, num_nodes_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_ones_, ones_.data(), num_nodes_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    check_cuda(cudaMalloc(&d_db_, bias_expanded_.num_columns_ * sizeof(float)));

    check_cuda(cudaMalloc(&d_weight_, weight_.size_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_weight_, weight_.values_, weight_.size_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    check_cuda(cudaMalloc(&d_dx_, gradients_input_.size_ * sizeof(float)));

    check_cuda(cudaMalloc(&d_dweight_, grad_weight_.num_rows_ * grad_weight_.num_columns_ * sizeof(float)));
}

// d_dx = linear_self_.backward_compute(d_dy, d_x);
float *Linear::backward_compute(float *d_dy, float *d_x) {
    float alpha = 1.0;
    float beta = 0.0;

    // dBias = incoming_gradients * ones
    check_cublas(cublasSgemv(cuda_helper_->cublas_handle,
                             CUBLAS_OP_T,
                             num_nodes_, num_out_features_,
                             &alpha, d_dy, num_nodes_,
                             d_ones_, 1,
                             &beta, d_db_, 1));

    // gradients_input = incoming_gradients * weight.T
    check_cublas(cublasSgemm(cuda_helper_->cublas_handle,
                             CUBLAS_OP_N, CUBLAS_OP_T,
                             num_nodes_, weight_.num_rows_, num_out_features_,
                             &alpha,
                             d_dy, num_nodes_,
                             d_weight_, weight_.num_rows_,
                             &beta,
                             d_dx_, num_nodes_));

    // dWeight = input.T * incoming_gradients
    check_cublas(cublasSgemm(cuda_helper_->cublas_handle,
                             CUBLAS_OP_T, CUBLAS_OP_N,
                             num_in_features_, num_out_features_, num_nodes_,
                             &alpha,
                             d_x, num_nodes_,
                             d_dy, num_nodes_,
                             &beta,
                             d_dweight_, grad_weight_.num_rows_));

    return d_dx_;
}

void Linear::backward_free() {
    check_cuda(cudaFree(d_ones_));
    check_cuda(cudaFree(d_db_));
    check_cuda(cudaFree(d_weight_));
    check_cuda(cudaFree(d_dx_));
    check_cuda(cudaFree(d_dweight_));
}

Matrix<float> *Linear::backward(Matrix<float> *incoming_gradients) {
    to_column_major_inplace(incoming_gradients);
    to_column_major_inplace(x_);
    to_column_major_inplace(&weight_);
    to_column_major_inplace(&bias_);

    backward_init();

    float *d_dy;
    check_cuda(cudaMalloc(&d_dy, incoming_gradients->size_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_dy, incoming_gradients->values_, incoming_gradients->size_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    float *d_x;
    check_cuda(cudaMalloc(&d_x, x_->size_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_x, x_->values_, x_->size_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    backward_compute(d_dy, d_x);

    // gradients of bias
    check_cuda(cudaMemcpy(grad_bias_.values_, d_db_, grad_bias_.size_ * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // gradients of input
    check_cuda(cudaMemcpy(gradients_input_.values_, d_dx_, gradients_input_.size_ * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // gradients of weight
    check_cuda(cudaMemcpy(grad_weight_.values_, d_dweight_, grad_weight_.size_ * sizeof(float),
                          cudaMemcpyDeviceToHost));

    backward_free();

    return &gradients_input_;
}
