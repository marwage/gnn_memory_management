// Copyright 2020 Marcel Wagenländer

#include "axpby.h"
#include "cuda_helper.hpp"

#include <cstdlib>
#include <iostream>
#include "catch2/catch.hpp"


int test_axpby(int num_elements) {
    float alpha = 2.0;
    float beta = 3.0;
    float *x = (float *) malloc(num_elements * sizeof(float));
    float *y = (float *) malloc(num_elements * sizeof(float));
    float *y_result = (float *) malloc(num_elements * sizeof(float));
    float *d_y_result = (float *) malloc(num_elements * sizeof(float));
    float *d_x;
    float *d_y;

    for (int i = 0; i < num_elements; ++i) {
        x[i] = rand();
        y[i] = rand();
    }

    for (int i = 0; i < num_elements; ++i) {
        y_result[i] = (alpha * x[i]) + (beta * y[i]);
    }

    check_cuda(cudaMalloc(&d_x, num_elements * sizeof(float)));
    check_cuda(cudaMemcpy(d_x, x, num_elements * sizeof(float), cudaMemcpyHostToDevice));
    check_cuda(cudaMalloc(&d_y, num_elements * sizeof(float)));
    check_cuda(cudaMemcpy(d_y, y, num_elements * sizeof(float), cudaMemcpyHostToDevice));

    xpy(alpha, d_x, beta, d_y, num_elements);

    check_cuda(cudaMemcpy(d_y_result, d_y, num_elements * sizeof(float), cudaMemcpyDeviceToHost));

    int equal = 1;
    for (int i = 0; i < num_elements; ++i) {
        if (y_result[i] != d_y_result[i]) {
            equal = 0;
        }
    }

    return equal;
}

TEST_CASE("a * x + b * y", "[axpby]") {
    CHECK(test_axpby(1e3));
    CHECK(test_axpby(1e4));
    CHECK(test_axpby(1e5));
    CHECK(test_axpby(1e6));
}
