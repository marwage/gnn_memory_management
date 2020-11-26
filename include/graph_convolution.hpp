// Copyright 2020 Marcel Wagenländer

#ifndef GRAPH_CONVOLUTION_H
#define GRAPH_CONVOLUTION_H

#include <vector>

#include "cuda_helper.hpp"
#include "tensors.hpp"


class GraphConvolutionParent {
public:
    virtual matrix<float>* forward(matrix<float> *x) = 0;
    virtual matrix<float>* backward(matrix<float> *in_gradients) = 0;
};

class GraphConvolution: public GraphConvolutionParent {
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
    GraphConvolution();
    GraphConvolution(CudaHelper *helper, sparse_matrix<float> *adjacency, std::string reduction,
                     long num_nodes, long num_features);
    matrix<float>* forward(matrix<float> *x);
    matrix<float>* forward(sparse_matrix<float> *adj, matrix<float> *x);
    matrix<float>* backward(matrix<float> *in_gradients);
};

class GraphConvChunked: public GraphConvolutionParent {
private:
    CudaHelper *cuda_helper_;
    long chunk_size_;
    long last_chunk_size_;
    long num_chunks_;
    std::vector<GraphConvolution> graph_conv_layers_;
    std::vector<sparse_matrix<float>> adjacencies_;
    matrix<float> y_;
    matrix<float> gradients_;
    std::vector<matrix<float>> x_chunks_;
    std::vector<matrix<float>> in_gradients_chunks_;

public:
    GraphConvChunked(CudaHelper *helper, sparse_matrix<float> *adjacency, std::string reduction,
                     long num_features, long chunk_size, long num_nodes);
    matrix<float>* forward(matrix<float> *x);
    matrix<float>* backward(matrix<float> *in_gradients);
};

#endif
