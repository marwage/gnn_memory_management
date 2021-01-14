// Copyright 2020 Marcel Wagenländer


#include <iostream>

#define SIZE 256

void add(float *a, float *b, int N) {
    for (int i = 0; i < N; i = i + 1) {
        a[i] = a[i] + b[i];
    }
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
    float *a = reinterpret_cast<float *>(malloc(SIZE * sizeof(float)));
    float *b = reinterpret_cast<float *>(malloc(SIZE * sizeof(float)));

    init_array(a, SIZE, 1);
    init_array(b, SIZE, 2);

    add(a, b, SIZE);

    print_array(a, SIZE);

    free(a);
    free(b);

    return 0;
}
