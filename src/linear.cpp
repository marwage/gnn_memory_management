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

    num_in_features_ = in_features;
    num_out_features_ = out_features;

    weight_.set(num_in_features_, num_out_features_, false);
    bias_.set(num_out_features_, 1, false);

    grad_weight_.set(weight_.num_rows_, weight_.num_columns_, false);
    grad_bias_.set(bias_.num_rows_, bias_.num_columns_, false);

    Linear::init_weight_bias();

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

Matrix<float> *Linear::forward(Matrix<float> *x) {
    to_column_major_inplace(x);
    x_ = x;
    to_column_major_inplace(&weight_);
    to_column_major_inplace(&bias_);

    float *d_x;
    float *d_weight;
    float *d_bias;
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_x),
                          x->num_rows_ * x->num_columns_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_x, x->values_,
                          x->num_rows_ * x->num_columns_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    check_cuda(cudaMalloc(&d_weight,
                          weight_.num_rows_ * weight_.num_columns_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_weight, weight_.values_,
                          weight_.num_rows_ * weight_.num_columns_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    Linear::expand_bias();
    check_cuda(cudaMalloc(&d_bias,
                          bias_expanded_.num_rows_ * bias_expanded_.num_columns_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_bias, bias_expanded_.values_,
                          bias_expanded_.num_rows_ * bias_expanded_.num_columns_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    float alpha = 1.0;
    float beta = 1.0;
    check_cublas(cublasSgemm(cuda_helper_->cublas_handle,// PyTorch uses GEMM too
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             x->num_rows_, weight_.num_columns_, x->num_columns_,
                             &alpha,
                             d_x, x->num_rows_,
                             d_weight, weight_.num_rows_,
                             &beta,
                             d_bias, x->num_rows_));

    // get result of linear
    check_cuda(cudaMemcpy(y_.values_, d_bias,
                          y_.num_rows_ * y_.num_columns_ * sizeof(float),
                          cudaMemcpyDeviceToHost));

    y_.is_row_major_ = false;

    // free GPU memory
    check_cuda(cudaFree(d_x));
    check_cuda(cudaFree(d_weight));
    check_cuda(cudaFree(d_bias));

    return &y_;
}

void Linear::backward(Matrix<float> *incoming_gradients, Matrix<float> *x, Matrix<float> *gradients) {
    to_column_major_inplace(incoming_gradients);
    to_column_major_inplace(x);
    to_column_major_inplace(&weight_);
    to_column_major_inplace(&bias_);

    float alpha = 1.0;
    float beta = 0.0;

    // gradients of bias
    float *d_g;
    check_cuda(cudaMalloc(&d_g, incoming_gradients->size_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_g, incoming_gradients->values_,
                          incoming_gradients->size_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    float *d_ones;
    check_cuda(cudaMalloc(&d_ones, incoming_gradients->num_rows_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_ones, ones_.data(),
                          incoming_gradients->num_rows_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    float *d_db;
    check_cuda(cudaMalloc(&d_db, incoming_gradients->num_columns_ * sizeof(float)));


    check_cublas(cublasSgemv(cuda_helper_->cublas_handle,
                             CUBLAS_OP_T,
                             incoming_gradients->num_rows_, incoming_gradients->num_columns_,
                             &alpha, d_g, incoming_gradients->num_rows_,
                             d_ones, 1,
                             &beta, d_db, 1));

    check_cuda(cudaMemcpy(grad_bias_.values_, d_db,
                          grad_bias_.num_rows_ * grad_bias_.num_columns_ * sizeof(float),
                          cudaMemcpyDeviceToHost));

    check_cuda(cudaFree(d_ones));
    check_cuda(cudaFree(d_db));

    // gradient of weight
    // gradients_input = in_gradients * weight.T
    float *d_weight;
    check_cuda(cudaMalloc(&d_weight, weight_.size_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_weight, weight_.values_,
                          weight_.size_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    float *d_dinput;
    check_cuda(cudaMalloc(&d_dinput,
                          gradients->size_ * sizeof(float)));

    check_cublas(cublasSgemm(cuda_helper_->cublas_handle,
                             CUBLAS_OP_N, CUBLAS_OP_T,
                             incoming_gradients->num_rows_, weight_.num_rows_, incoming_gradients->num_columns_,
                             &alpha,
                             d_g, incoming_gradients->num_rows_,
                             d_weight, weight_.num_rows_,
                             &beta,
                             d_dinput, gradients->num_rows_));

    check_cuda(cudaMemcpy(gradients->values_, d_dinput,
                          gradients->num_rows_ * gradients->num_columns_ * sizeof(float),
                          cudaMemcpyDeviceToHost));

    check_cuda(cudaFree(d_dinput));
    check_cuda(cudaFree(d_weight));

    // dWeight = input.T * in_gradients
    float *d_input;
    check_cuda(cudaMalloc(&d_input, x->size_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_input, x->values_,
                          x->num_rows_ * x->num_columns_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    float *d_dweight;
    check_cuda(cudaMalloc(&d_dweight, grad_weight_.num_rows_ * grad_weight_.num_columns_ * sizeof(float)));

    check_cublas(cublasSgemm(cuda_helper_->cublas_handle,
                             CUBLAS_OP_T, CUBLAS_OP_N,
                             x->num_columns_, incoming_gradients->num_columns_, x->num_rows_,
                             &alpha,
                             d_input, x->num_rows_,
                             d_g, incoming_gradients->num_rows_,
                             &beta,
                             d_dweight, grad_weight_.num_rows_));

    check_cuda(cudaMemcpy(grad_weight_.values_, d_dweight,
                          grad_weight_.num_rows_ * grad_weight_.num_columns_ * sizeof(float),
                          cudaMemcpyDeviceToHost));

    check_cuda(cudaFree(d_g));
    check_cuda(cudaFree(d_dweight));
    check_cuda(cudaFree(d_input));
}

Matrix<float> *Linear::backward(Matrix<float> *in_gradients) {
    backward(in_gradients, x_, &gradients_input_);

    return &gradients_input_;
}
