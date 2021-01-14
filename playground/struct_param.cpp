// Copyright 2020 Marcel Wagenländer

#include <iostream>


struct holder {
    int N;
    float *ptr;
};

class Klasse {
public:
    holder holder_;

    Klasse();
    void func_a(holder hol);
    void func_b();
};

Klasse::Klasse() {}

void Klasse::func_a(holder hol) {
    holder_ = hol;
    std::cout << holder_.ptr << std::endl;
    for (int i = 0; i < holder_.N; ++i) {
        std::cout << holder_.ptr[i] << std::endl;
    }
}

void Klasse::func_b() {
    std::cout << holder_.ptr << std::endl;
    for (int i = 0; i < holder_.N; ++i) {
        std::cout << holder_.ptr[i] << std::endl;
    }
}

int main() {
    int N = 4;
    float *arr = (float *) malloc(N * sizeof(float));
    for (int i = 0; i < N; ++i) {
        arr[i] = (float) i;
    }
    holder hol;
    hol.N = N;
    hol.ptr = arr;

    Klasse klasse;

    klasse.func_a(hol);
    klasse.func_b();
}
