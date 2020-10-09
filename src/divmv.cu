// Copyright 2020 Marcel Wagenländer

__global__
void divmv(float *X, float*y, int n, int m) {
    // n rows, i is blockIdx.x, blockDim.x is rows
    // m columns, j is threadIdx.x,
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * m) X[idx] = X[idx] / y[blockIdx.x];
}

void div_mat_vec(float *X, float*y, int n, int m) {
    divmv<<<n, m>>>(X, y, n, m);
}
