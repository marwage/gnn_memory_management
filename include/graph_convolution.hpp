// Copyright 2020 Marcel Wagenländer

#ifndef GRAPH_CONVOLUTION_H
#define GRAPH_CONVOLUTION_H

#include <vector>

#include "cuda_helper.hpp"
#include "tensors.hpp"


class GraphConvolution {
private:
    CudaHelper *cuda_helper_ = NULL;
    SparseMatrix<float> *adjacency_ = NULL;
    std::string reduction_;
    bool mean_;
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
    Matrix<float> *backward(Matrix<float> *in_gradients);
};

class GraphConvChunked {
private:
    CudaHelper *cuda_helper_ = NULL;
    long chunk_size_;
    long last_chunk_size_;
    long num_chunks_;
    long num_nodes_;
    bool mean_;
    SparseMatrix<float> *adjacency_ = NULL;
    std::vector<SparseMatrix<float>> adjacencies_;
    Matrix<float> sum_;
    std::vector<Matrix<float>> y_;
    std::vector<Matrix<float>> gradients_;

public:
    GraphConvChunked(CudaHelper *helper, SparseMatrix<float> *adjacency, std::string reduction,
                     long num_features, long chunk_size, long num_nodes);
    std::vector<Matrix<float>> *forward(std::vector<Matrix<float>> *x);
    std::vector<Matrix<float>> *backward(std::vector<Matrix<float>> *incoming_gradients);
    void forward(std::vector<Matrix<float>> *x, Matrix<float> *y);
    void backward(Matrix<float> *incoming_gradients, Matrix<float> *gradients);
};

#endif
