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

}

