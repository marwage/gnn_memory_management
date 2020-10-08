// Copyright 2020 Marcel Wagenländer

#include "linear.hpp"
#include "cuda_helper.hpp"

#include <random>


matrix<float> linear(matrix<float> X) {
    int num_hidden_channels = 8;
    matrix<float> weight;
    weight.rows = X.columns;
    weight.columns = num_hidden_channels;
    weight.values = (float *) malloc(weight.rows * weight.columns * sizeof(float));
    vector<float> bias;
    bias.size = num_hidden_channels;
    bias.values = (float *) malloc(bias.size * sizeof(float));

    // init weight and bias
    double k = 1.0 / (double) X.columns;
    k = sqrt(k);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distr(-k, k);
    for (int i = 0; i < weight.rows * weight.columns; ++i) {
        weight.values[i] = distr(gen);
    }
    for (int i = 0; i < bias.size; ++i) {
        bias.values[i] = distr(gen);
    }

    cudaError_t cuda_error;
    cublasStatus_t cublas_status;
    cublasHandle_t cublas_handle;
    cublas_status = cublasCreate(&cublas_handle);
    check_cublas(cublas_status);

    float *d_X, *d_weight, *d_bias;
    matrix<float> X_col;
    X_col.rows = X.columns;
    X_col.columns = X.rows;
    X_col.values = (float *) malloc(X_col.rows * X_col.columns * sizeof(float));
    transpose(X_col.values, X.values, X.rows, X.columns);
    cuda_error = cudaMalloc((void **) &d_X, X.rows * X.columns * sizeof(float));
    check_cuda(cuda_error);
    cuda_error = cudaMemcpy(d_X, X_col.values, X_col.rows * X_col.columns * sizeof(float),
            cudaMemcpyHostToDevice);
    matrix<float> weight_col;
    weight_col.rows = weight.columns;
    weight_col.columns = weight.rows;
    weight_col.values = (float *) malloc(weight_col.rows * weight_col.columns * sizeof(float));
    transpose(weight_col.values, weight.values, weight.rows, weight.columns);
    cuda_error = cudaMalloc(&d_weight, weight_col.rows * weight_col.columns * sizeof(float));
    check_cuda(cuda_error);
    cuda_error = cudaMemcpy(d_weight, weight_col.values,
            weight_col.rows * weight_col.columns * sizeof(float),
            cudaMemcpyHostToDevice);
    check_cuda(cuda_error);
    matrix<float> bias_expanded;
    bias_expanded.rows = X.rows;
    bias_expanded.columns = bias.size;
    bias_expanded.values = (float *) malloc(bias_expanded.rows * bias_expanded.columns * sizeof(float));
    for (int i = 0; i < X.rows; ++i) {
        std::memcpy(&bias_expanded.values[i * bias.size],
                bias.values,
                bias.size * sizeof(float));
    }
    matrix<float> bias_expanded_col;
    bias_expanded_col.rows = bias_expanded.columns;
    bias_expanded_col.columns = bias_expanded.rows;
    bias_expanded_col.values = (float *) malloc(bias_expanded_col.rows * bias_expanded_col.columns * sizeof(float));
    transpose(bias_expanded_col.values, bias_expanded.values,
            bias_expanded.rows, bias_expanded.columns);
    cuda_error = cudaMalloc(&d_bias,
            bias_expanded_col.rows * bias_expanded_col.columns * sizeof(float));
    check_cuda(cuda_error);
    cuda_error = cudaMemcpy(d_bias, bias_expanded_col.values,
            bias_expanded_col.rows * bias_expanded_col.columns * sizeof(float),
            cudaMemcpyHostToDevice);
    check_cuda(cuda_error);
    
    float alpha = 1.0;
    float beta = 1.0;
    // PyTorch uses GEMM too
    cublas_status = cublasSgemm(cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            X.rows, num_hidden_channels, X.columns,
            &alpha,
            d_X, X.rows,
            d_weight, weight.rows,
            &beta,
            d_bias, X.rows);
    check_cublas(cublas_status);

    // get result of linear
    matrix<float> result_col;
    result_col.rows = bias_expanded_col.rows;
    result_col.columns = bias_expanded_col.columns;
    result_col.values = (float *) malloc(result_col.rows * result_col.columns * sizeof(float));
    cuda_error = cudaMemcpy(result_col.values, d_bias,
            result_col.rows * result_col.columns * sizeof(float),
            cudaMemcpyDeviceToHost);
    check_cuda(cuda_error);
    matrix<float> result;
    result.rows = result_col.columns;
    result.columns = result_col.rows;
    result.values = (float *) malloc(result.rows * result.columns * sizeof(float));
    transpose(result.values, result_col.values, result_col.rows, result_col.columns);

    // free GPU memory
    cuda_error = cudaFree(d_X);
    check_cuda(cuda_error);
    cuda_error = cudaFree(d_weight);
    check_cuda(cuda_error);
    cuda_error = cudaFree(d_bias);
    check_cuda(cuda_error);

    // free CPU memory
    free(weight.values);
    free(bias.values);
    free(X_col.values);
    free(weight_col.values);
    free(bias_expanded.values);
    free(bias_expanded_col.values);
    free(result_col.values);
    
    // clean cuBLAS
    cublas_status = cublasDestroy(cublas_handle);

    return result;
}
