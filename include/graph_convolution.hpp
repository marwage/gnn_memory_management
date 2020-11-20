// Copyright 2020 Marcel Wagenländer

#ifndef GRAPH_CONVOLUTION_H
#define GRAPH_CONVOLUTION_H

#include <vector>

#include "cuda_helper.hpp"
#include "tensors.hpp"


class GraphConvolution {
private:
    CudaHelper *cuda_helper_;
    sparse_matrix<float> *adjacency_;
    std::string reduction_;
    bool mean_;
    matrix<float> ones_;
    matrix<float> sum_;
    matrix<float> y_;
    matrix<float> gradients_;

public:
    GraphConvolution(CudaHelper *helper, sparse_matrix<float> *adjacency, std::string reduction, long num_features);
    matrix<float> forward(matrix<float> x);
    matrix<float> forward(sparse_matrix<float> *adj, matrix<float> B);
    matrix<float> backward(matrix<float> in_gradients);
};

class GraphConvChunked {
private:
    CudaHelper *cuda_helper_;
    std::vector<GraphConvolution> graph_conv_layers_;
    int chunk_size_;
    int last_chunk_size_;
    int num_chunks_;

public:
    GraphConvChunked(CudaHelper *helper, std::string reduction, int chunk_size, int num_nodes);
    matrix<float> forward(sparse_matrix<float> adj, matrix<float> B);
    matrix<float> backward(matrix<float> in_gradients);
};

#endif
