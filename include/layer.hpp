// Copyright 2020 Marcel Wagenländer

#include "tensors.hpp"


class Layer {
public:
    Matrix<float> *forward(Matrix<float> *x);
    Matrix<float> *backward(Matrix<float> *in_gradients);
};
