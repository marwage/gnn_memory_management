// Copyright 2021 Marcel Wagenländer

#include "sage_convolution.hpp"
#include "dense_computation.hpp"

SAGEConvolution::SAGEConvolution() {}

SAGEConvolution::SAGEConvolution(CudaHelper *helper, long num_nodes, long num_in_features, long num_out_features,
                                 SparseMatrix<float> *adjacency, AggregationReduction reduction, Matrix<float> *adjacency_row_sum) {
    SAGEConvolution::set(helper, num_nodes, num_in_features, num_out_features,
                         adjacency, reduction, adjacency_row_sum);
}

void SAGEConvolution::set(CudaHelper *helper, long num_nodes, long num_in_features, long num_out_features,
                          SparseMatrix<float> *adjacency, AggregationReduction reduction, Matrix<float> *adjacency_row_sum) {
    cuda_helper_ = helper;

    linear_self_.set(helper, num_in_features, num_out_features, num_nodes);
    linear_neighbourhood_.set(helper, num_in_features, num_out_features, num_nodes);
    feature_aggregation_.set(helper, num_nodes, num_in_features, adjacency, reduction, adjacency_row_sum);
    add_.set(helper, num_nodes, num_out_features);
}

Matrix<float> *SAGEConvolution::forward(Matrix<float> *x) {
    Matrix<float> *neighbourhood_features = feature_aggregation_.forward(x);
    Matrix<float> *neighbourhood_activations = linear_neighbourhood_.forward(neighbourhood_features);
    Matrix<float> *self_activations = linear_self_.forward(x);
    return add_.forward(neighbourhood_activations, self_activations);
}

Matrix<float> *SAGEConvolution::backward(Matrix<float> *incoming_gradients) {
    std::vector<Matrix<float> *> *add_gradients = add_.backward(incoming_gradients);
    Matrix<float> *self_gradients = linear_self_.backward(add_gradients->at(1));
    Matrix<float> *neighbourhood_gradients = linear_neighbourhood_.backward(add_gradients->at(0));
    Matrix<float> *aggregation_gradients = feature_aggregation_.backward(neighbourhood_gradients);

    mat_mat_add(cuda_helper_, aggregation_gradients, self_gradients, &gradients_);

    return &gradients_;
}
