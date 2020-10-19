// Copyright 2020 Marcel Wagenländer

#include <cstdlib>
#include <iostream>

#include "axpby.h"
#include "cuda_helper.hpp"


int check_axpby(int num_elements) {
    float alpha = 2.0;
    float beta = 3.0;
    float x[num_elements];
    float y[num_elements];
    float y_result[num_elements];
    float d_y_result[num_elements];
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

    bool equal = true;
    for (int i = 0; i < num_elements; ++i) {
        if (y_result[i] != d_y_result[i]) {
            equal = false;
        }
    }

    if (equal) {
        std::cout << "axpby works" << std::endl;
    } else {
        std::cout << "axpby does not works" << std::endl;
    }
}

int main() {
    for (int i = 1; i < 100; ++i) {
        std::cout << i << std::endl;
        check_axpby(i * 1000);
    }
}