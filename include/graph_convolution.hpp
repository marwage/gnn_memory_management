// Copyright 2020 Marcel Wagenländer

#ifndef GRAPH_CONVOLUTION_H
#define GRAPH_CONVOLUTION_H

#include <vector>

#include "cuda_helper.hpp"
#include "tensors.hpp"


class GraphConvolution {
private:
    CudaHelper *cuda_helper_;
    SparseMatrix<float> *adjacency_;
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
protected:
    CudaHelper *cuda_helper_;
    long chunk_size_;
    long last_chunk_size_;
    long num_chunks_;
    bool mean_;
    SparseMatrix<float> *adjacency_;
    std::vector<SparseMatrix<float>> adjacencies_;
    Matrix<float> sum_;
    std::vector<Matrix<float>> y_;
    std::vector<Matrix<float>> gradients_;

public:
    GraphConvChunked();
    GraphConvChunked(CudaHelper *helper, SparseMatrix<float> *adjacency, std::string reduction,
                     long num_features, long chunk_size, long num_nodes);
    virtual void set(CudaHelper *helper, SparseMatrix<float> *adjacency, std::string reduction,
                     long num_features, long chunk_size, long num_nodes);
    virtual std::vector<Matrix<float>> *forward(std::vector<Matrix<float>> *x);
    virtual std::vector<Matrix<float>> *backward(std::vector<Matrix<float>> *incoming_gradients);
};

class GraphConvPipelined : public GraphConvChunked {
protected:
    long num_steps_;
    float *d_y_;
    float *d_sum_forward_;
    float *d_gradients_;
    std::vector<float *> d_x_;
    std::vector<SparseMatrixCuda<float>> d_adj_;
    std::vector<float *> d_incoming_gradients_;
    std::vector<float *> d_sum_backward_;

public:
    GraphConvPipelined();
    GraphConvPipelined(CudaHelper *helper, SparseMatrix<float> *adjacency, std::string reduction,
                       long num_features, long chunk_size, long num_nodes);
    void set(CudaHelper *helper, SparseMatrix<float> *adjacency, std::string reduction,
             long num_features, long chunk_size, long num_nodes) override;
    std::vector<Matrix<float>> *forward(std::vector<Matrix<float>> *x) override;
    std::vector<Matrix<float>> *backward(std::vector<Matrix<float>> *incoming_gradients) override;
};

#endif
