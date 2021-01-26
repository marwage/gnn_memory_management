// Copyright 2021 Marcel Wagenländer

#ifndef ALZHEIMER_SAGE_CONVOLUTION_H
#define ALZHEIMER_SAGE_CONVOLUTION_H

#include "add.hpp"
#include "feature_aggregation.hpp"
#include "linear.hpp"


class SAGEConvolution : Layer {
protected:
    CudaHelper *cuda_helper_;
    Matrix<float> gradients_;
    FeatureAggregation feature_aggregation_;
    Linear linear_self_;
    Linear linear_neighbourhood_;
    Add add_;

public:
    SAGEConvolution();
    SAGEConvolution(CudaHelper *helper, long num_nodes, long num_in_features, long num_out_features,
                    SparseMatrix<float> *adjacency, AggregationReduction reduction, Matrix<float> *adjacency_row_sum);
    void set(CudaHelper *helper, long num_nodes, long num_in_features, long num_out_features,
             SparseMatrix<float> *adjacency, AggregationReduction reduction, Matrix<float> *adjacency_row_sum);
    Matrix<float> *forward(Matrix<float> *x) override;
    Matrix<float> *backward(Matrix<float> *incoming_gradients) override;
};

#endif//ALZHEIMER_SAGE_CONVOLUTION_H
