// Copyright 2020 Marcel Wagenländer

#include <cuda_runtime.h>
#include <random>
#include <chrono>

#include "cuda_helper.hpp"
#include "linear.hpp"
#include "tensors.hpp"


Linear::Linear() {}

Linear::Linear(CudaHelper *helper, long in_features, long out_features, long num_nodes) {
    cuda_helper_ = helper;

    num_in_features_ = in_features;
    num_out_features_ = out_features;

    weight_ = Matrix<float>(num_in_features_, num_out_features_, false);
    bias_ = Matrix<float>(num_out_features_, 1, false);

    grad_weight_ = Matrix<float>(weight_.rows, weight_.columns, false);
    grad_bias_= Matrix<float>(bias_.rows, bias_.columns, false);

    Linear::init_weight_bias();

    bias_expanded_ = Matrix<float>(num_nodes, bias_.rows, false);


    y_ = Matrix<float>(num_nodes, weight_.columns, false);

    ones_ = new float[num_nodes];
    for (int i = 0; i < num_nodes; ++i) {
        ones_[i] = 1.0;
    }

    gradients_input_ = Matrix<float>(num_nodes, in_features, false);
}

void Linear::init_weight_bias() {
    double k = 1.0 / static_cast<double>(num_in_features_);
    k = sqrt(k);
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_real_distribution<float> distr(-k, k);
    for (int i = 0; i < weight_.rows * weight_.columns; ++i) {
        weight_.values[i] = distr(generator);
    }
    for (int i = 0; i < bias_.rows * bias_.columns; ++i) {
        bias_.values[i] = distr(generator);
    }
}

Matrix<float>** Linear::get_parameters() {
    Matrix<float> **parameters = new Matrix<float>*[2];
    parameters[0] = &weight_;
    parameters[1] = &bias_;

    return parameters;
}

void Linear::set_parameters(Matrix<float> **parameters) {
    to_column_major_inplace(parameters[0]);
    to_column_major_inplace(parameters[1]);

    std::memcpy(weight_.values, parameters[0]->values, weight_.rows * weight_.columns * sizeof(float));
    std::memcpy(bias_.values, parameters[1]->values, bias_.rows * bias_.columns * sizeof(float));
}

Matrix<float>** Linear::get_gradients() {
    Matrix<float> **grads = new Matrix<float>*[2];
    grads[0] = &grad_weight_;
    grads[1] = &grad_bias_;

    return grads;
}

void Linear::set_gradients(Matrix<float> **grads) {
    to_column_major_inplace(grads[0]);
    to_column_major_inplace(grads[1]);

    std::memcpy(grad_weight_.values, grads[0]->values, grad_weight_.rows * grad_weight_.columns * sizeof(float));
    std::memcpy(grad_bias_.values, grads[1]->values, grad_bias_.rows * grad_bias_.columns * sizeof(float));
}

void Linear::expand_bias() {
    for (int i = 0; i < bias_expanded_.columns; ++i) {
        for (int j = 0; j < bias_expanded_.rows; ++j) {
            bias_expanded_.values[i * bias_expanded_.rows + j] = bias_.values[i];
        }
    }
}

Matrix<float>* Linear::forward(Matrix<float> *x) {
    if (y_.rows != x->rows || num_in_features_ != x->columns) {
        throw "Matrix shapes unequal";
    }
    to_column_major_inplace(x);
    x_ = x;
    to_column_major_inplace(&weight_);
    to_column_major_inplace(&bias_);

    float *d_X, *d_weight, *d_bias;
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_X),
                          x->rows * x->columns * sizeof(float)));
    check_cuda(cudaMemcpy(d_X, x->values,
                          x->rows * x->columns * sizeof(float),
                          cudaMemcpyHostToDevice));

    check_cuda(cudaMalloc(&d_weight,
                          weight_.rows * weight_.columns * sizeof(float)));
    check_cuda(cudaMemcpy(d_weight, weight_.values,
                          weight_.rows * weight_.columns * sizeof(float),
                          cudaMemcpyHostToDevice));

    Linear::expand_bias();
    check_cuda(cudaMalloc(&d_bias,
                          bias_expanded_.rows * bias_expanded_.columns * sizeof(float)));
    check_cuda(cudaMemcpy(d_bias, bias_expanded_.values,
                          bias_expanded_.rows * bias_expanded_.columns * sizeof(float),
                          cudaMemcpyHostToDevice));

    float alpha = 1.0;
    float beta = 1.0;
    check_cublas(cublasSgemm(cuda_helper_->cublas_handle,// PyTorch uses GEMM too
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             x->rows, weight_.columns, x->columns,
                             &alpha,
                             d_X, x->rows,
                             d_weight, weight_.rows,
                             &beta,
                             d_bias, x->rows));

    // get result of linear
    check_cuda(cudaMemcpy(y_.values, d_bias,
                          y_.rows * y_.columns * sizeof(float),
                          cudaMemcpyDeviceToHost));

    y_.row_major = false;

    // free GPU memory
    check_cuda(cudaFree(d_X));
    check_cuda(cudaFree(d_weight));
    check_cuda(cudaFree(d_bias));

    return &y_;
}

Matrix<float>* Linear::backward(Matrix<float> *in_gradients) {
    to_column_major_inplace(in_gradients);
    to_column_major_inplace(x_);
    to_column_major_inplace(&weight_);
    to_column_major_inplace(&bias_);

    float alpha = 1.0;
    float beta = 0.0;

    // gradients of bias
    float *d_g;
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_g),
                          in_gradients->rows * in_gradients->columns * sizeof(float)));
    check_cuda(cudaMemcpy(d_g, in_gradients->values,
                          in_gradients->rows * in_gradients->columns * sizeof(float),
                          cudaMemcpyHostToDevice));

    float *d_ones;
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_ones),
                          in_gradients->rows * sizeof(float)));
    check_cuda(cudaMemcpy(d_ones, ones_,
                          in_gradients->rows * sizeof(float),
                          cudaMemcpyHostToDevice));

    float *d_db;
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_db),
                          in_gradients->columns * sizeof(float)));


    check_cublas(cublasSgemv(cuda_helper_->cublas_handle,
                             CUBLAS_OP_T,
                             in_gradients->rows, in_gradients->columns,
                             &alpha, d_g, in_gradients->rows,
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
                             in_gradients->rows, weight_.rows, in_gradients->columns,
                             &alpha,
                             d_g, in_gradients->rows,
                             d_weight, weight_.rows,
                             &beta,
                             d_dinput, gradients_input_.rows));

    check_cuda(cudaMemcpy(gradients_input_.values, d_dinput,
                          gradients_input_.rows * gradients_input_.columns * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // dWeight = input.T * in_gradients
    float *d_input;
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_input),
                          x_->rows * x_->columns * sizeof(float)));
    check_cuda(cudaMemcpy(d_input, x_->values,
                          x_->rows * x_->columns * sizeof(float),
                          cudaMemcpyHostToDevice));

    float *d_dweight;
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_dweight),
                          grad_weight_.rows * grad_weight_.columns * sizeof(float)));

    check_cublas(cublasSgemm(cuda_helper_->cublas_handle,
                             CUBLAS_OP_T, CUBLAS_OP_N,
                             x_->columns, in_gradients->columns, x_->rows,
                             &alpha,
                             d_input, x_->rows,
                             d_g, in_gradients->rows,
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

    return &gradients_input_;
}

void Linear::update_weights(Matrix<float> *gradients) {
    Matrix<float> *in_gradients_weight = &gradients[0];
    Matrix<float> *in_gradients_bias = &gradients[1];
    to_column_major_inplace(in_gradients_weight);
    to_column_major_inplace(in_gradients_bias);

    float *d_grads;
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_grads),
                          in_gradients_weight->rows * in_gradients_weight->columns * sizeof(float)));
    check_cuda(cudaMemcpy(d_grads, in_gradients_weight->values,
                          in_gradients_weight->rows * in_gradients_weight->columns * sizeof(float),
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
                          in_gradients_bias->rows * in_gradients_bias->columns * sizeof(float)));
    check_cuda(cudaMemcpy(d_grads, in_gradients_bias->values,
                          in_gradients_bias->rows * in_gradients_bias->columns * sizeof(float),
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
