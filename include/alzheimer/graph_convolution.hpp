// Copyright 2020 Marcel Wagenländer

#ifndef GRAPH_CONVOLUTION_H
#define GRAPH_CONVOLUTION_H

#include <iostream>
#include <vector>

#include "cuda_helper.hpp"
#include "tensors.hpp"


class GraphConvolution {
private:
    CudaHelper *cuda_helper_;
    sparse_matrix<float> *adjacency_;
    std::string reduction_;
    bool mean_;
    matrix<float> sum_;

public:
    GraphConvolution(CudaHelper *helper, sparse_matrix<float> *adjacency_, std::string reduction);
    matrix<float> forward(matrix<float> B);
    matrix<float> backward(matrix<float> in_gradients);
};

class GraphConvChunked {
private:
    CudaHelper *cuda_helper_;
    sparse_matrix<float> *adjacency_;
    std::string reduction_;
    bool mean_;
    matrix<float> sum_;
    GraphConvolution *graph_conv_layers_;
    int chunk_size_;
    int last_chunk_size_;
    int num_chunks_;
public:
    GraphConvChunked(CudaHelper *helper, sparse_matrix<float> *adjacency_, std::string reduction, int chunk_size);
    matrix<float> forward(matrix<float> B);
    matrix<float> backward(matrix<float> in_gradients);
};

#endif
