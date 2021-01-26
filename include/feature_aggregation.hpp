// Copyright 2020 Marcel Wagenländer

#ifndef FEATURE_AGGREGATION_H
#define FEATURE_AGGREGATION_H

#include "cuda_helper.hpp"
#include "layer.hpp"
#include "tensors.hpp"

#include <vector>

enum AggregationReduction { sum,
                            mean };

class FeatureAggregation : public Layer {
private:
    CudaHelper *cuda_helper_;
    SparseMatrix<float> *adjacency_;
    Matrix<float> *adjacency_row_sum_;
    AggregationReduction reduction_;
    Matrix<float> y_;
    Matrix<float> gradients_;

public:
    FeatureAggregation();
    FeatureAggregation(CudaHelper *helper, long num_nodes, long num_features,
                       SparseMatrix<float> *adjacency, AggregationReduction reduction, Matrix<float> *adjacency_row_sum);
    void set(CudaHelper *helper, long num_nodes, long num_features,
             SparseMatrix<float> *adjacency, AggregationReduction reduction, Matrix<float> *adjacency_row_sum);
    Matrix<float> *forward(Matrix<float> *x) override;
    Matrix<float> *backward(Matrix<float> *incoming_gradients) override;
};

class FeatureAggregationChunked {
protected:
    CudaHelper *cuda_helper_;
    long chunk_size_;
    long last_chunk_size_;
    long num_chunks_;
    bool mean_;
    std::vector<SparseMatrix<float>> *adjacencies_;
    Matrix<float> *adjacency_row_sum_;
    std::vector<Matrix<float>> y_;
    std::vector<Matrix<float>> gradients_;

    bool keep_allocation_;
    float *d_x_;
    float *d_y_;
    float *d_sum_;

    void allocate_gpu_memory();
    void free_gpu_memory();

public:
    std::string name_;

    FeatureAggregationChunked();
    FeatureAggregationChunked(CudaHelper *helper, std::vector<SparseMatrix<float>> *adjacencies, Matrix<float> *sum,
                              std::string reduction, long num_features, long chunk_size, long num_nodes);
    ~FeatureAggregationChunked();
    virtual void set(CudaHelper *helper, std::vector<SparseMatrix<float>> *adjacencies, Matrix<float> *sum,
                     std::string reduction, long num_features, long chunk_size, long num_nodes);
    virtual void set(CudaHelper *helper, std::vector<SparseMatrix<float>> *adjacencies, Matrix<float> *sum,
                     std::string reduction, long num_features, long chunk_size, long num_nodes, bool keep_allocation);
    virtual std::vector<Matrix<float>> *forward(std::vector<Matrix<float>> *x);
    virtual std::vector<Matrix<float>> *backward(std::vector<Matrix<float>> *incoming_gradients);
};

class FeatureAggregationPipelined : public FeatureAggregationChunked {
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
    FeatureAggregationPipelined();
    FeatureAggregationPipelined(CudaHelper *helper, std::vector<SparseMatrix<float>> *adjacencies, Matrix<float> *sum,
                                std::string reduction, long num_features, long chunk_size, long num_nodes);
    void set(CudaHelper *helper, std::vector<SparseMatrix<float>> *adjacencies, Matrix<float> *sum,
             std::string reduction, long num_features, long chunk_size, long num_nodes) override;
    std::vector<Matrix<float>> *forward(std::vector<Matrix<float>> *x) override;
    std::vector<Matrix<float>> *backward(std::vector<Matrix<float>> *incoming_gradients) override;
};

#endif//FEATURE_AGGREGATION_H
