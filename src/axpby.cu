// Copyright 2020 Marcel Wagenländer

#include <math.h>


__global__ void axpby(float alpha, float *x, float beta, float *y, int num_elements) {
    // n rows, i is blockIdx.x
    // m columns, j is threadIdx.x
    int idx = blockIdx.x + threadIdx.x;
    if (idx < num_elements) y[idx] = (alpha * x[idx]) + (beta * y[idx]);
}

void xpy(float alpha, float *x, float beta, float *y, int num_elements) {
    int num_threads = 1024;
    int num_blocks = ceil(num_elements / num_threads);
    axpby<<<num_blocks, num_threads>>>(alpha, x, beta, y, num_elements);
}
