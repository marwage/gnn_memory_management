// Copyright 2020 Marcel Wagenländer

__global__ void divmv(float *X, float *y, int n, int m) {
    // n rows, i is blockIdx.x
    // m columns, j is threadIdx.x
    int idx = threadIdx.x * n + blockIdx.x;
    if (idx < n * m && y[blockIdx.x] != 0.0) {
        X[idx] = X[idx] / y[blockIdx.x];
    }
}

void div_mat_vec(float *X, float *y, int n, int m) {
    divmv<<<n, m>>>(X, y, n, m);
}
