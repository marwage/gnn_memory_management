// Copyright 2020 Marcel Wagenländer

#include <iostream>


class Float {
public:
    float number_;
    Float(float number);
};

Float::Float(float number) {
    number_ = number;
}

int main() {
    int N = 10;
    Float *numbers = (Float *) malloc(N * sizeof(Float));
    for (int i = 0; i < N; ++i) {
        numbers[i] = Float((float) i);
    }
    for (int i = 0; i < N; ++i) {
        std::cout << "Number " << i << " " << numbers[i].number_ << std::endl;
    }
}

