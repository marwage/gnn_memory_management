// Copyright 2020 Marcel Wagenländer

#include <iostream>
#include <cmath>

#include "cuda_helper.hpp"
#include "invsqrt.h"


void check_invsqrt(int num_elements) {
    float epsilon = 1e-8;
    float x[num_elements];
    float x_result[num_elements];
    float *d_x;
    float d_x_result[num_elements];

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

    bool equal = true;
    int num_nans = 0;
    for (int i = 0; i < num_elements; ++i) {
        if (x_result[i] != d_x_result[i]) {
            equal = false;
        }
        if (std::isnan(d_x_result[i])) {
            num_nans = num_nans + 1;
        }
    }

    if (equal) {
        std::cout << "invsqrt works" << std::endl;
    } else {
        std::cout << "invsqrt does not works" << std::endl;
    }

    std::cout << "Number of NaNs " << num_nans << std::endl;
}

int main() {
    for (int i = 0; i < 100; ++i) {
        std::cout << i << std::endl;
        check_invsqrt(i * 1000);
    }
}