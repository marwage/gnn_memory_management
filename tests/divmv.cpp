// Copyright 2020 Marcel Wagenländer

#include <cuda_runtime.h>
#include <iostream>

#include "divmv.h"
#include "cuda_helper.hpp"
#include "tensors.hpp"


void check_divmv() {
    float *d_X, *d_y;
    int n = 5;
    int m = 7;

    check_cuda(cudaMalloc((void **) &d_X, n * m * sizeof(float)));
    check_cuda(cudaMalloc((void **) &d_y, n * sizeof(float)));

    float X[n * m];
    for (int i = 0; i < n * m; ++i) {
        X[i] = 1.0;
    }
    float y[n];
    for (int i = 0; i < n; ++i) {
        y[i] = (float) (i + 1);
    }

    float Z[n * m];
    for (int i = 0; i < n * m; ++i) {
        Z[i] = 1.0;
    }

    int idx;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            idx = j * n + i;
            Z[idx] = Z[idx] / y[i];
        }
    }
    matrix<float> Z_mat;
    Z_mat.rows = n;
    Z_mat.columns = m;
    Z_mat.values = Z;

    check_cuda(cudaMemcpy(d_X, X, n * m * sizeof(float),
                          cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(d_y, y, n * sizeof(float),
                          cudaMemcpyHostToDevice));

    div_mat_vec(d_X, d_y, n, m);

    check_cuda(cudaMemcpy(X, d_X, n * m * sizeof(float),
                          cudaMemcpyDeviceToHost));

    check_cuda(cudaFree(d_X));
    check_cuda(cudaFree(d_y));

    bool equal = true;
    for (int i = 0; i < Z_mat.rows * Z_mat.columns; ++i) {
        if (Z_mat.values[i] != Z_mat.values[i]) {
            equal = false;
            break;
        }
    }
    if (equal) {
        std::cout << "divmv works" << std::endl;
    } else {
        std::cout << "divmv does not work" << std::endl;
    }
}

int main() {
    check_divmv();
}
