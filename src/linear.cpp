// Copyright 2020 Marcel Wagenländer

#include <random>

#include "linear.hpp"
#include "cuda_helper.hpp"


Linear::Linear() { }

Linear::Linear(int in_features, int out_features) {
    num_in_features = in_features;
    num_out_features = out_features;

    weight.rows = num_in_features;
    weight.columns = num_out_features;
    bias.size = num_out_features;

    Linear::init_weight_bias();
}

void Linear::init_weight_bias() {
    weight.values = reinterpret_cast<float *>(
            malloc(weight.rows * weight.columns * sizeof(float)));
    bias.values = reinterpret_cast<float *>(
            malloc(bias.size * sizeof(float)));

    double k = 1.0 / static_cast<double>(num_out_features);
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
}

matrix<float>* Linear::get_parameters() {
    matrix<float> *parameters = (matrix<float> *) malloc(2 * sizeof(matrix<float>));
    parameters[0] = weight;
    matrix<float> bias_mat;
    bias_mat.rows = bias.size;
    bias_mat.columns = 1;
    bias_mat.values = bias.values;
    parameters[1] = bias_mat;

    return parameters;
}

matrix<float> Linear::expand_bias(int num_rows) {
    matrix<float> bias_expanded;
    bias_expanded.rows = num_rows;
    bias_expanded.columns = bias.size;
    bias_expanded.values = reinterpret_cast<float *>(
            malloc(bias_expanded.rows * bias_expanded.columns * sizeof(float)));
    for (int i = 0; i < num_rows; ++i) {
        std::memcpy(&bias_expanded.values[i * bias.size],
                bias.values, bias.size * sizeof(float));
    }

    return bias_expanded;
}

// assume X is column-major
matrix<float> Linear::forward(matrix<float> X) {
    cudaError_t cuda_error;
    cublasStatus_t cublas_status;
    cublasHandle_t cublas_handle;
    cublas_status = cublasCreate(&cublas_handle);
    check_cublas(cublas_status);

    float *d_X, *d_weight, *d_bias;
    cuda_error = cudaMalloc(reinterpret_cast<void **>(&d_X),
            X.rows * X.columns * sizeof(float));
    check_cuda(cuda_error);
    cuda_error = cudaMemcpy(d_X, X.values, X.rows * X.columns * sizeof(float),
            cudaMemcpyHostToDevice);

    cuda_error = cudaMalloc(&d_weight,
            weight.rows * weight.columns * sizeof(float));
    check_cuda(cuda_error);
    cuda_error = cudaMemcpy(d_weight, weight.values,
            weight.rows * weight.columns * sizeof(float),
            cudaMemcpyHostToDevice);
    check_cuda(cuda_error);

    matrix<float> bias_expanded = Linear::expand_bias(X.rows);
    cuda_error = cudaMalloc(&d_bias,
            bias_expanded.rows * bias_expanded.columns * sizeof(float));
    check_cuda(cuda_error);
    cuda_error = cudaMemcpy(d_bias, bias_expanded.values,
            bias_expanded.rows * bias_expanded.columns * sizeof(float),
            cudaMemcpyHostToDevice);
    check_cuda(cuda_error);

    float alpha = 1.0;
    float beta = 1.0;
    cublas_status = cublasSgemm(cublas_handle,  // PyTorch uses GEMM too
            CUBLAS_OP_N, CUBLAS_OP_N,
            X.rows, weight.columns, X.columns,
            &alpha,
            d_X, X.rows,
            d_weight, weight.rows,
            &beta,
            d_bias, X.rows);
    check_cublas(cublas_status);

    // get result of linear
    matrix<float> result;
    result.rows = X.rows;
    result.columns = weight.columns;
    result.values = reinterpret_cast<float *>(
            malloc(result.rows * result.columns * sizeof(float)));
    cuda_error = cudaMemcpy(result.values, d_bias,
            result.rows * result.columns * sizeof(float),
            cudaMemcpyDeviceToHost);
    check_cuda(cuda_error);

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
    free(bias_expanded.values);

    // clean cuBLAS
    cublas_status = cublasDestroy(cublas_handle);

    return result;  // column-major
}

