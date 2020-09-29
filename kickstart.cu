// Copyright 2020 Marcel Wagenländer


#include <iostream>

#define SIZE 256

__global__ void add(float *a, float *b, int N) {
    int i = threadIdx.x;
    a[i] = a[i] + b[i];
}

void print_array(float *a, int N) {
    for (int i = 0; i < N; i = i + 1) {
        std::cout << a[i] << "\n";
    }
}

void init_array(float *a, int N, float value) {
    for (int i = 0; i < N; i = i + 1) {
        a[i] = value * i;
    }
}


int main() {
    float *a, *b;
    
    cudaMallocManaged(&a, SIZE * sizeof(float));
    cudaMallocManaged(&b, SIZE * sizeof(float));

    init_array(a, SIZE, 1);
    init_array(b, SIZE, 2);

    add<<<1, SIZE>>>(a, b, SIZE);

    cudaDeviceSynchronize();

    print_array(a, SIZE);

    cudaFree(a);
    cudaFree(b);

    return 0;
}
