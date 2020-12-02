// Copyright 2020 Marcel Wagenländer

#include "tensors.hpp"

#include <iostream>


class Outer {
    public:
    Matrix<float> mat_;
    Outer();
};

Outer::Outer() {
    mat_.set(512, 256, true);

    std::cout << "Size: " << mat_.size_ << std::endl;
    std::cout << "Pointer: " << mat_.values_ << std::endl;
    std::cout << "Value 512: " << mat_.values_[512] << std::endl;
}

int main() {
    Outer outer = Outer();

    std::cout << "Value 1024: " << outer.mat_.values_[1024] << std::endl;
}
