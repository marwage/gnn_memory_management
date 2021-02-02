// Copyright 2020 Marcel Wagenländer

#include <math.h>

#include "axdy.cuh"


__global__ void axdy(float alpha, float *x, float *y, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) y[idx] = alpha * x[idx] * y[idx];
}

void ax_dot_y(float alpha, float *x, float *y, int num_elements) {
    int num_threads = 1024;
    int num_blocks = ceil((float) num_elements / (float) num_threads);
    axdy<<<num_blocks, num_threads>>>(alpha, x, y, num_elements);
}
