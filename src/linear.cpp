// Copyright 2020 Marcel Wagenländer

#include <cuda_runtime.h>
#include <random>

#include "cuda_helper.hpp"
#include "linear.hpp"
#include "tensors.hpp"


Linear::Linear() {}

Linear::Linear(CudaHelper *helper, long in_features, long out_features, long num_nodes) {
    cuda_helper_ = helper;

    num_in_features_ = in_features;
    num_out_features_ = out_features;

    weight_ = new_float_matrix(num_in_features_, num_out_features_, false);
    bias_ = new_float_matrix(num_out_features_, 1, false);

    grad_weight_ = new_float_matrix(weight_.rows, weight_.columns, false);
    grad_bias_= new_float_matrix(bias_.rows, bias_.columns, false);

    Linear::init_weight_bias();

    bias_expanded_ = new_float_matrix(num_nodes, bias_.rows, false);


    y_ = new_float_matrix(num_nodes, weight_.columns, false);

    ones_ = new float[num_nodes];
    for (int i = 0; i < num_nodes; ++i) {
        ones_[i] = 1.0;
    }

    gradients_input_ = new_float_matrix(num_nodes, in_features, false);
}

void Linear::init_weight_bias() {
    double k = 1.0 / static_cast<double>(num_in_features_);
    k = sqrt(k);
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distr(-k, k);
    for (int i = 0; i < weight_.rows * weight_.columns; ++i) {
        weight_.values[i] = distr(generator);
    }
    for (int i = 0; i < bias_.rows * bias_.columns; ++i) {
        bias_.values[i] = distr(generator);
    }
}

matrix<float> *Linear::get_parameters() {
    matrix<float> *parameters = new matrix<float>[2];
    parameters[0] = weight_;
    parameters[1] = bias_;

    return parameters;
}

void Linear::set_parameters(matrix<float> *parameters) {
    weight_ = parameters[0];
    to_column_major_inplace(&weight_);
    bias_ = parameters[1];
    to_column_major_inplace(&bias_);
}

matrix<float> *Linear::get_gradients() {
    matrix<float> *grads = new matrix<float>[2];
    grads[0] = grad_weight_;
    grads[1] = grad_bias_;

    return grads;
}

void Linear::set_gradients(matrix<float> *grads) {
    grad_weight_ = grads[0];
    to_column_major_inplace(&grad_weight_);
    grad_bias_ = grads[1];
    to_column_major_inplace(&grad_bias_);
}

matrix<float> Linear::expand_bias() {
    for (int i = 0; i < bias_expanded_.columns; ++i) {
        for (int j = 0; j < bias_expanded_.rows; ++j) {
            bias_expanded_.values[i * bias_expanded_.rows + j] = bias_.values[i];
        }
    }

    return bias_expanded_;
}

matrix<float> Linear::forward(matrix<float> X) {
    if (X.rows < 1) {
        throw "Input to Linear::forward has a non-positive number of rows";
    }
    to_column_major_inplace(&X);
    x_ = X;

    float *d_X, *d_weight, *d_bias;
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_X),
                          X.rows * X.columns * sizeof(float)));
    check_cuda(cudaMemcpy(d_X, X.values,
                          X.rows * X.columns * sizeof(float),
                          cudaMemcpyHostToDevice));

    check_cuda(cudaMalloc(&d_weight,
                          weight_.rows * weight_.columns * sizeof(float)));
    check_cuda(cudaMemcpy(d_weight, weight_.values,
                          weight_.rows * weight_.columns * sizeof(float),
                          cudaMemcpyHostToDevice));

    matrix<float> bias_expanded = Linear::expand_bias();
    check_cuda(cudaMalloc(&d_bias,
                          bias_expanded.rows * bias_expanded.columns * sizeof(float)));
    check_cuda(cudaMemcpy(d_bias, bias_expanded.values,
                          bias_expanded.rows * bias_expanded.columns * sizeof(float),
                          cudaMemcpyHostToDevice));

    float alpha = 1.0;
    float beta = 1.0;
    check_cublas(cublasSgemm(cuda_helper_->cublas_handle,// PyTorch uses GEMM too
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             X.rows, weight_.columns, X.columns,
                             &alpha,
                             d_X, X.rows,
                             d_weight, weight_.rows,
                             &beta,
                             d_bias, X.rows));

    // get result of linear
    check_cuda(cudaMemcpy(y_.values, d_bias,
                          y_.rows * y_.columns * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // free GPU memory
    check_cuda(cudaFree(d_X));
    check_cuda(cudaFree(d_weight));
    check_cuda(cudaFree(d_bias));

    return y_;
}

matrix<float> Linear::backward(matrix<float> in_gradients) {
    to_column_major_inplace(&in_gradients);

    float alpha = 1.0;
    float beta = 0.0;

    // gradients of bias
    float *d_g;
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_g),
                          in_gradients.rows * in_gradients.columns * sizeof(float)));
    check_cuda(cudaMemcpy(d_g, in_gradients.values,
                          in_gradients.rows * in_gradients.columns * sizeof(float),
                          cudaMemcpyHostToDevice));

    float *d_ones;
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_ones),
                          in_gradients.rows * sizeof(float)));
    check_cuda(cudaMemcpy(d_ones, ones_,
                          in_gradients.rows * sizeof(float),
                          cudaMemcpyHostToDevice));

    float *d_db;
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_db),
                          in_gradients.columns * sizeof(float)));


    check_cublas(cublasSgemv(cuda_helper_->cublas_handle,
                             CUBLAS_OP_T,
                             in_gradients.rows, in_gradients.columns,
                             &alpha, d_g, in_gradients.rows,
                             d_ones, 1,
                             &beta, d_db, 1));

    check_cuda(cudaMemcpy(grad_bias_.values, d_db,
                          grad_bias_.rows * grad_bias_.columns * sizeof(float),
                          cudaMemcpyDeviceToHost));

    check_cuda(cudaFree(d_ones));
    check_cuda(cudaFree(d_db));

    // gradient of weight
    // gradients_input = in_gradients * weight.T
    float *d_weight;
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_weight),
                          weight_.rows * weight_.columns * sizeof(float)));
    check_cuda(cudaMemcpy(d_weight, weight_.values,
                          weight_.rows * weight_.columns * sizeof(float),
                          cudaMemcpyHostToDevice));

    float *d_dinput;
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_dinput),
                          gradients_input_.rows * gradients_input_.columns * sizeof(float)));

    check_cublas(cublasSgemm(cuda_helper_->cublas_handle,
                             CUBLAS_OP_N, CUBLAS_OP_T,
                             in_gradients.rows, weight_.rows, in_gradients.columns,
                             &alpha,
                             d_g, in_gradients.rows,
                             d_weight, weight_.rows,
                             &beta,
                             d_dinput, gradients_input_.rows));

    check_cuda(cudaMemcpy(gradients_input_.values, d_dinput,
                          gradients_input_.rows * gradients_input_.columns * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // dWeight = input.T * in_gradients
    float *d_input;
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_input),
                          x_.rows * x_.columns * sizeof(float)));
    check_cuda(cudaMemcpy(d_input, x_.values,
                          x_.rows * x_.columns * sizeof(float),
                          cudaMemcpyHostToDevice));

    float *d_dweight;
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_dweight),
                          grad_weight_.rows * grad_weight_.columns * sizeof(float)));

    check_cublas(cublasSgemm(cuda_helper_->cublas_handle,
                             CUBLAS_OP_T, CUBLAS_OP_N,
                             x_.columns, in_gradients.columns, x_.rows,
                             &alpha,
                             d_input, x_.rows,
                             d_g, in_gradients.rows,
                             &beta,
                             d_dweight, grad_weight_.rows));

    check_cuda(cudaMemcpy(grad_weight_.values, d_dweight,
                          grad_weight_.rows * grad_weight_.columns * sizeof(float),
                          cudaMemcpyDeviceToHost));

    check_cuda(cudaFree(d_g));
    check_cuda(cudaFree(d_weight));
    check_cuda(cudaFree(d_dweight));
    check_cuda(cudaFree(d_input));
    check_cuda(cudaFree(d_dinput));

    return gradients_input_;
}

void Linear::update_weights(matrix<float> *gradients) {
    grad_weight_ = gradients[0];
    grad_bias_ = gradients[1];
    to_column_major_inplace(&grad_weight_);
    to_column_major_inplace(&grad_bias_);

    float *d_grads;
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_grads),
                          grad_weight_.rows * grad_weight_.columns * sizeof(float)));
    check_cuda(cudaMemcpy(d_grads, grad_weight_.values,
                          grad_weight_.rows * grad_weight_.columns * sizeof(float),
                          cudaMemcpyHostToDevice));

    float *d_weight;
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_weight),
                          weight_.rows * weight_.columns * sizeof(float)));
    check_cuda(cudaMemcpy(d_weight, weight_.values,
                          weight_.rows * weight_.columns * sizeof(float),
                          cudaMemcpyHostToDevice));

    float alpha = -1.0;
    check_cublas(cublasSaxpy(cuda_helper_->cublas_handle,
                             weight_.rows * weight_.columns,
                             &alpha, d_grads, 1,
                             d_weight, 1));

    check_cuda(cudaMemcpy(weight_.values, d_weight,
                          weight_.rows * weight_.columns * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // clean-up
    check_cuda(cudaFree(d_grads));
    check_cuda(cudaFree(d_weight));

    check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_grads),
                          grad_bias_.rows * grad_bias_.columns * sizeof(float)));
    check_cuda(cudaMemcpy(d_grads, grad_bias_.values,
                          grad_bias_.rows * grad_bias_.columns * sizeof(float),
                          cudaMemcpyHostToDevice));

    float *d_bias;
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_bias),
                          bias_.rows * bias_.columns * sizeof(float)));
    check_cuda(cudaMemcpy(d_bias, bias_.values,
                          bias_.rows * bias_.columns * sizeof(float),
                          cudaMemcpyHostToDevice));

    check_cublas(cublasSaxpy(cuda_helper_->cublas_handle,
                             bias_.rows * bias_.columns,
                             &alpha, d_grads, 1,
                             d_bias, 1));

    check_cuda(cudaMemcpy(bias_.values, d_bias,
                          bias_.rows * bias_.columns * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // clean-up
    check_cuda(cudaFree(d_grads));
    check_cuda(cudaFree(d_bias));
}
