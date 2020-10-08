// Copyright 2020 Marcel Wagenländer

#ifndef GRAPH_CONVOLUTION_H
#define GRAPH_CONVOLUTION_H

#include <iostream>

#include "tensors.hpp"


matrix<float> graph_convolution(sparse_matrix<float> A, matrix<float> B,
        std::string reduction);

matrix<float> graph_convolution_debug(sparse_matrix<float> A, matrix<float> B);

#endif
