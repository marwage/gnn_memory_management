// Copyright 2020 Marcel Wagenländer

#include "invsqrt.cuh"
#include "cuda_helper.hpp"

#include "catch2/catch.hpp"
#include <cmath>
#include <iostream>


int test_invsqrt(int num_elements) {
    float epsilon = 1e-8;
    float *x = (float *) malloc(num_elements * sizeof(float));
    float *x_result = (float *) malloc(num_elements * sizeof(float));
    float *d_x_result = (float *) malloc(num_elements * sizeof(float));
    float *d_x;

    for (int i = 0; i < num_elements; ++i) {
        x[i] = rand();
    }

    for (int i = 0; i < num_elements; ++i) {
        x_result[i] = 1 / (sqrtf(x[i]) + epsilon);
    }

    check_cuda(cudaMalloc(&d_x, num_elements * sizeof(float)));
    check_cuda(cudaMemcpy(d_x, x, num_elements * sizeof(float), cudaMemcpyHostToDevice));

    inverse_sqrt(d_x, epsilon, num_elements);

    check_cuda(cudaMemcpy(d_x_result, d_x, num_elements * sizeof(float), cudaMemcpyDeviceToHost));

    int equal = 1;
    int num_nans = 0;
    for (int i = 0; i < num_elements; ++i) {
        if (x_result[i] != d_x_result[i]) {
            equal = 0;
        }
        if (std::isnan(d_x_result[i])) {
            num_nans = num_nans + 1;
        }
    }

    std::cout << "Number of NaNs " << num_nans << std::endl;

    return equal;
}


TEST_CASE("Inverse square root", "[invsqrt]") {
    CHECK(test_invsqrt(1e3));
    CHECK(test_invsqrt(1e4));
    CHECK(test_invsqrt(1e5));
    CHECK(test_invsqrt(1e6));
}
