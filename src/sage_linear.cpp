// Copyright 2020 Marcel Wagenländer

#include "sage_linear.hpp"
#include "cuda_helper.hpp"


SageLinear::SageLinear(int in_features, int out_features) {
    num_in_features_ = in_features;
    num_out_features_ = out_features;
    linear_self_ = Linear(num_in_features_, num_out_features_);
    linear_neigh_= Linear(num_in_features_, num_out_features_);
}

matrix<float> SageLinear::forward(matrix<float> features,
        matrix<float> aggr) {
    matrix<float> self_result = linear_self_.forward(features);
    matrix<float> neigh_result = linear_neigh_.forward(aggr);

    cudaError_t cuda_error;
    cublasStatus_t cublas_status;
    cublasHandle_t cublas_handle;
    cublas_status = cublasCreate(&cublas_handle);
    check_cublas(cublas_status);

    float *d_self;
    cuda_error = cudaMalloc((void **) &d_self,
            self_result.rows * self_result.columns * sizeof(float));
    check_cuda(cuda_error);
    cuda_error = cudaMemcpy(d_self, self_result.values,
            self_result.rows * self_result.columns * sizeof(float),
            cudaMemcpyHostToDevice);
    check_cuda(cuda_error);

    float *d_neigh;
    cuda_error = cudaMalloc((void **) &d_neigh,
            neigh_result.rows * neigh_result.columns * sizeof(float));
    check_cuda(cuda_error);
    cuda_error = cudaMemcpy(d_neigh, neigh_result.values,
            neigh_result.rows * neigh_result.columns * sizeof(float),
            cudaMemcpyHostToDevice);
    check_cuda(cuda_error);

    float alpha = 1.0;
    cublas_status = cublasSaxpy(cublas_handle,
            self_result.rows * self_result.columns, &alpha,
            d_neigh, 1,
            d_self, 1);
    check_cublas(cublas_status);

    cuda_error = cudaMemcpy(self_result.values, d_self,
            self_result.rows * self_result.columns * sizeof(float),
            cudaMemcpyDeviceToHost);
    check_cuda(cuda_error);

    cuda_error = cudaFree(d_self);
    check_cuda(cuda_error);
    cuda_error = cudaFree(d_neigh);
    check_cuda(cuda_error);

    cublas_status = cublasDestroy(cublas_handle);
    check_cublas(cublas_status);

    return self_result;
}

