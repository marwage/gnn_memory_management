// Copyright 2020 Marcel Wagenländer

#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "tensors.hpp"


matrix<float> relu(matrix<float> X);

matrix<float> softmax(matrix<float> X);

#endif
