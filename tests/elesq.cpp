// Copyright 2020 Marcel Wagenländer

#include <iostream>
#include <cmath>

#include "cuda_helper.hpp"
#include "elesq.h"


void check_elesq(int num_elements) {
    float x[num_elements];
    float x_result[num_elements];
    float *d_x;
    float d_x_result[num_elements];

    for (int i = 0; i < num_elements; ++i) {
        x[i] = rand();
    }

    for (int i = 0; i < num_elements; ++i) {
        x_result[i] = x[i] * x[i];
    }

    check_cuda(cudaMalloc(&d_x, num_elements * sizeof(float)));
    check_cuda(cudaMemcpy(d_x, x, num_elements * sizeof(float), cudaMemcpyHostToDevice));

    ele_squared(d_x, num_elements);

    check_cuda(cudaMemcpy(d_x_result, d_x, num_elements * sizeof(float), cudaMemcpyDeviceToHost));

    bool equal = true;
    for (int i = 0; i < num_elements; ++i) {
        if (x_result[i] != d_x_result[i]) {
            equal = false;
        }
    }

    if (equal) {
        std::cout << "elesq works" << std::endl;
    } else {
        std::cout << "elesq does not works" << std::endl;
    }
}

int main() {
    for (int i = 0; i < 100; ++i) {
        std::cout << i << std::endl;
        check_elesq(i * 1000);
    }
}