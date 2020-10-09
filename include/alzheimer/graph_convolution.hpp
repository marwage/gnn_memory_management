// Copyright 2020 Marcel Wagenländer

#ifndef GRAPH_CONVOLUTION_H
#define GRAPH_CONVOLUTION_H

#include <iostream>

#include "tensors.hpp"
#include "cuda_helper.hpp"


class GraphConvolution {
private:
    CudaHelper *cuda_helper_;
public:
    GraphConvolution(CudaHelper *helper);
    matrix<float> forward(sparse_matrix<float> A, matrix<float> B,
                          std::string reduction);
};

#endif
