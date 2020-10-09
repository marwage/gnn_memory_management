// Copyright 2020 Marcel Wagenländer

#include <random>
#include <cstring>

#include "linear.hpp"
#include "cuda_helper.hpp"


Linear::Linear() { }

Linear::Linear(int in_features, int out_features, CudaHelper *helper) {
    cuda_helper_ = helper;

    num_in_features_ = in_features;
    num_out_features_ = out_features;

    weight_.rows = num_in_features_;
    weight_.columns = num_out_features_;
    bias_.rows = num_out_features_;
    bias_.columns = 1;

    Linear::init_weight_bias();
}

void Linear::init_weight_bias() {
    weight_.values = reinterpret_cast<float *>(
            malloc(weight_.rows * weight_.columns * sizeof(float)));
    bias_.values = reinterpret_cast<float *>(
            malloc(bias_.rows * bias_.columns * sizeof(float)));

    double k = 1.0 / static_cast<double>(num_out_features_);
    k = sqrt(k);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distr(-k, k);
    for (int i = 0; i < weight_.rows * weight_.columns; ++i) {
        weight_.values[i] = distr(gen);
    }
    for (int i = 0; i < bias_.rows * bias_.columns; ++i) {
        bias_.values[i] = distr(gen);
    }
}

matrix<float>* Linear::get_parameters() {
    matrix<float> *parameters = (matrix<float> *) malloc(2 * sizeof(matrix<float>));
    parameters[0] = weight_;
    parameters[1] = bias_;

    return parameters;
}

matrix<float> Linear::expand_bias(int num_rows) {
    matrix<float> bias_expanded;
    bias_expanded.rows = num_rows;
    bias_expanded.columns = bias_.rows;
    bias_expanded.values = reinterpret_cast<float *>(
            malloc(bias_expanded.rows * bias_expanded.columns * sizeof(float)));
    for (int i = 0; i < num_rows; ++i) {
        std::memcpy(&bias_expanded.values[i * bias_.rows],
                    bias_.values, bias_.rows * sizeof(float));
    }

    return bias_expanded;
}

// assume X is column-major
matrix<float> Linear::forward(matrix<float> X) {
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

    matrix<float> bias_expanded = Linear::expand_bias(X.rows);
    check_cuda(cudaMalloc(&d_bias,
                          bias_expanded.rows * bias_expanded.columns * sizeof(float)));
    check_cuda(cudaMemcpy(d_bias, bias_expanded.values,
                          bias_expanded.rows * bias_expanded.columns * sizeof(float),
                          cudaMemcpyHostToDevice));

    float alpha = 1.0;
    float beta = 1.0;
    check_cublas(cublasSgemm(cuda_helper_->cublas_handle,  // PyTorch uses GEMM too
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             X.rows, weight_.columns, X.columns,
                             &alpha,
                             d_X, X.rows,
                             d_weight, weight_.rows,
                             &beta,
                             d_bias, X.rows));

    // get result of linear
    matrix<float> result;
    result.rows = X.rows;
    result.columns = weight_.columns;
    result.values = reinterpret_cast<float *>(
            malloc(result.rows * result.columns * sizeof(float)));
    check_cuda(cudaMemcpy(result.values, d_bias,
                          result.rows * result.columns * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // free GPU memory
    check_cuda(cudaFree(d_X));
    check_cuda(cudaFree(d_weight));
    check_cuda(cudaFree(d_bias));

    // free CPU memory
    free(weight_.values);
    free(bias_.values);
    free(bias_expanded.values);

    return result;
}

