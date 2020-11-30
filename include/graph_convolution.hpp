// Copyright 2020 Marcel Wagenländer

#ifndef GRAPH_CONVOLUTION_H
#define GRAPH_CONVOLUTION_H

#include <vector>

#include "cuda_helper.hpp"
#include "tensors.hpp"


class GraphConvolutionParent {
public:
    virtual Matrix<float> *forward(Matrix<float> *x) = 0;
    virtual Matrix<float> *backward(Matrix<float> *in_gradients) = 0;
};

class GraphConvolution : public GraphConvolutionParent {
private:
    CudaHelper *cuda_helper_ = NULL;
    SparseMatrix<float> *adjacency_ = NULL;
    std::string reduction_;
    bool mean_;
    Matrix<float> ones_;
    Matrix<float> sum_;
    Matrix<float> y_;
    Matrix<float> gradients_;

public:
    GraphConvolution();
    GraphConvolution(CudaHelper *helper, SparseMatrix<float> *adjacency, std::string reduction,
                     long num_nodes, long num_features);
    void set(CudaHelper *helper, SparseMatrix<float> *adjacency, std::string reduction,
             long num_nodes, long num_features);
    Matrix<float> *forward(Matrix<float> *x);
    Matrix<float> *forward(SparseMatrix<float> *adj, Matrix<float> *x);
    Matrix<float> *backward(Matrix<float> *in_gradients);
};

class GraphConvChunked : public GraphConvolutionParent {
private:
    CudaHelper *cuda_helper_ = NULL;
    long chunk_size_;
    long last_chunk_size_;
    long num_chunks_;
    std::vector<GraphConvolution> graph_conv_layers_;
    std::vector<SparseMatrix<float>> adjacencies_;
    Matrix<float> y_;
    Matrix<float> gradients_;
    std::vector<Matrix<float>> x_chunks_;
    std::vector<Matrix<float>> in_gradients_chunks_;

public:
    GraphConvChunked(CudaHelper *helper, SparseMatrix<float> *adjacency, std::string reduction,
                     long num_features, long chunk_size, long num_nodes);
    Matrix<float> *forward(Matrix<float> *x);
    Matrix<float> *backward(Matrix<float> *in_gradients);
};

#endif
