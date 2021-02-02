// Copyright 2020 Marcel Wagenländer

#include <math.h>

#include "elesq.cuh"


__global__ void elesq(float *x, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) x[idx] = x[idx] * x[idx];
}

void ele_squared(float *x, int num_elements) {
    int num_threads = 1024;
    int num_blocks = ceil((float) num_elements / (float) num_threads);
    elesq<<<num_blocks, num_threads>>>(x, num_elements);
}
