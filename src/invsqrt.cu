// Copyright 2020 Marcel Wagenländer

#include <math.h>

#include "elesq.cuh"


__global__ void invsqrt(float *x, float epsilon, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) x[idx] = 1 / (sqrtf(x[idx]) + epsilon);
}

void inverse_sqrt(float *x, float epsilon, int num_elements) {
    int num_threads = 1024;
    int num_blocks = ceil((float) num_elements / (float) num_threads);
    invsqrt<<<num_blocks, num_threads>>>(x, epsilon, num_elements);
}
