// Copyright 2020 Marcel Wagenländer

#include "cuda_helper.hpp"
#include "tensors.hpp"

#include "catch2/catch.hpp"
#include <cuda_runtime.h>


Matrix<float> sum_rows(Matrix<float> in_mat) {
    CudaHelper cuda_helper;
    float alpha = 1.0;
    float beta = 0.0;

    float *d_mat;
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_mat),
                          in_mat.num_rows_ * in_mat.num_columns_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_mat, in_mat.values_,
                          in_mat.num_rows_ * in_mat.num_columns_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    Matrix<float> ones;
    ones.num_rows_ = in_mat.num_rows_;
    ones.num_columns_ = 1;
    ones.values_ = reinterpret_cast<float *>(malloc(ones.num_rows_ * sizeof(float)));
    for (int i = 0; i < ones.num_rows_; ++i) {
        ones.values_[i] = 1.0;
    }
    float *d_ones;
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_ones),
                          ones.num_rows_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_ones, ones.values_,
                          ones.num_rows_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    Matrix<float> sum;
    sum.num_rows_ = in_mat.num_columns_;
    sum.num_columns_ = 1;
    sum.values_ = reinterpret_cast<float *>(malloc(sum.num_rows_ * sizeof(float)));
    float *d_sum;
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_sum),
                          sum.num_rows_ * sizeof(float)));


    check_cublas(cublasSgemv(cuda_helper.cublas_handle,
                             CUBLAS_OP_T,
                             in_mat.num_rows_, in_mat.num_columns_,
                             &alpha, d_mat, in_mat.num_rows_,
                             d_ones, 1,
                             &beta, d_sum, 1));

    check_cuda(cudaMemcpy(sum.values_, d_sum,
                          sum.num_rows_ * sizeof(float),
                          cudaMemcpyDeviceToHost));

    check_cuda(cudaFree(d_mat));
    check_cuda(cudaFree(d_ones));
    check_cuda(cudaFree(d_sum));

    return sum;
}

int test_sum_rows() {
    Matrix<float> mat;
    mat.num_rows_ = 1 << 15;
    mat.num_columns_ = 1 << 13;
    mat.values_ = (float *) malloc(mat.num_rows_ * mat.num_columns_ * sizeof(float));
    for (int i = 0; i < mat.num_rows_ * mat.num_columns_; ++i) {
        mat.values_[i] = 2.0;
    }

    Matrix<float> sum = sum_rows(mat);

    float expected_value = (float) mat.num_rows_ * 2;
    for (int i = 0; i < sum.num_rows_; ++i) {
        if (sum.values_[i] != expected_value) {
            return 0;
        }
    }

    return 1;
}

TEST_CASE("Sum rows", "[sumrows]") {
    CHECK(test_sum_rows());
}
