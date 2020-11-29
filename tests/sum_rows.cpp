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
                          in_mat.rows * in_mat.columns * sizeof(float)));
    check_cuda(cudaMemcpy(d_mat, in_mat.values,
                          in_mat.rows * in_mat.columns * sizeof(float),
                          cudaMemcpyHostToDevice));

    Matrix<float> ones;
    ones.rows = in_mat.rows;
    ones.columns = 1;
    ones.values = reinterpret_cast<float *>(malloc(ones.rows * sizeof(float)));
    for (int i = 0; i < ones.rows; ++i) {
        ones.values[i] = 1.0;
    }
    float *d_ones;
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_ones),
                          ones.rows * sizeof(float)));
    check_cuda(cudaMemcpy(d_ones, ones.values,
                          ones.rows * sizeof(float),
                          cudaMemcpyHostToDevice));

    Matrix<float> sum;
    sum.rows = in_mat.columns;
    sum.columns = 1;
    sum.values = reinterpret_cast<float *>(malloc(sum.rows * sizeof(float)));
    float *d_sum;
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_sum),
                          sum.rows * sizeof(float)));


    check_cublas(cublasSgemv(cuda_helper.cublas_handle,
                             CUBLAS_OP_T,
                             in_mat.rows, in_mat.columns,
                             &alpha, d_mat, in_mat.rows,
                             d_ones, 1,
                             &beta, d_sum, 1));

    check_cuda(cudaMemcpy(sum.values, d_sum,
                          sum.rows * sizeof(float),
                          cudaMemcpyDeviceToHost));

    check_cuda(cudaFree(d_mat));
    check_cuda(cudaFree(d_ones));
    check_cuda(cudaFree(d_sum));

    return sum;
}

int test_sum_rows() {
    Matrix<float> mat;
    mat.rows = 1 << 15;
    mat.columns = 1 << 13;
    mat.values = (float *) malloc(mat.rows * mat.columns * sizeof(float));
    for (int i = 0; i < mat.rows * mat.columns; ++i) {
        mat.values[i] = 2.0;
    }

    Matrix<float> sum = sum_rows(mat);

    float expected_value = (float) mat.rows * 2;
    for (int i = 0; i < sum.rows; ++i) {
        if (sum.values[i] != expected_value) {
            return 0;
        }
    }

    return 1;
}

TEST_CASE("Sum rows", "[sumrows]") {
    CHECK(test_sum_rows());
}
