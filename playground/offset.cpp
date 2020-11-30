// Copyright 2020 Marcel Wagenländer

#include <iostream>


int main() {
    int N = 10;
    float *arr = (float *) malloc(N * sizeof(float));
    for (int i = 0; i < N; ++i) {
        arr[i] = (float) i;
    }
    for (int i = 0; i < N; ++i) {
        std::cout << "Number " << i << ": " << arr[i] << std::endl;
    }
    float *ptr = &arr[N / 2];
    for (int i = 0; i < N / 2; ++i) {
        ptr[i] = (float) i * 7;
    }
    for (int i = 0; i < N; ++i) {
        std::cout << "Number " << i << ": " << arr[i] << std::endl;
    }
}
