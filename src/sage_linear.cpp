// Copyright 2020 Marcel Wagenländer

#include "sage_linear.hpp"
#include "dense_computation.hpp"

#include <cmath>
#include <string>


std::vector<Matrix<float> *> SageLinearParent::get_parameters() {
    std::vector<Matrix<float> *> self_params = linear_self_.get_parameters();
    std::vector<Matrix<float> *> neigh_params = linear_neigh_.get_parameters();
    std::vector<Matrix<float> *> parameters(4);
    parameters[0] = self_params[0];
    parameters[1] = self_params[1];
    parameters[2] = neigh_params[0];
    parameters[3] = neigh_params[1];

    return parameters;
}

std::vector<Matrix<float> *> SageLinearParent::get_gradients() {
    std::vector<Matrix<float> *> self_grads = linear_self_.get_gradients();
    std::vector<Matrix<float> *> neigh_grads = linear_neigh_.get_gradients();
    std::vector<Matrix<float> *> gradients(4);
    gradients[0] = self_grads[0];
    gradients[1] = self_grads[1];
    gradients[2] = neigh_grads[0];
    gradients[3] = neigh_grads[1];

    return gradients;
}

SageLinear::SageLinear() {}

SageLinear::SageLinear(CudaHelper *helper, long in_features, long out_features, long num_nodes) {
    set(helper, in_features, out_features, num_nodes);
}

void SageLinear::set(CudaHelper *helper, long in_features, long out_features, long num_nodes) {
    cuda_helper_ = helper;

    num_in_features_ = in_features;
    num_out_features_ = out_features;

    linear_self_.set(cuda_helper_, num_in_features_, num_out_features_, num_nodes);
    linear_neigh_.set(cuda_helper_, num_in_features_, num_out_features_, num_nodes);
    y_.set(num_nodes, num_out_features_, false);
}

Matrix<float> *SageLinear::forward(Matrix<float> *features, Matrix<float> *aggr) {
    Matrix<float> *self_result = linear_self_.forward(features);
    Matrix<float> *neigh_result = linear_neigh_.forward(aggr);

    mat_mat_add(cuda_helper_, self_result, neigh_result, &y_);

    return &y_;
}

SageLinearGradients *SageLinear::backward(Matrix<float> *in_gradients) {
    input_gradients_.self_gradients = linear_self_.backward(in_gradients);
    input_gradients_.neighbourhood_gradients = linear_neigh_.backward(in_gradients);

    return &input_gradients_;
}

SageLinearChunked::SageLinearChunked(CudaHelper *helper, long num_in_features, long num_out_features, long chunk_size, long num_nodes) {
    cuda_helper_ = helper;
    chunk_size_ = chunk_size;
    num_in_features_ = num_in_features;
    num_out_features_ = num_out_features;

    linear_self_.set(cuda_helper_, num_in_features_, num_out_features_, chunk_size_);
    linear_neigh_.set(cuda_helper_, num_in_features_, num_out_features_, chunk_size_);

    num_chunks_ = ceil((float) num_nodes / (float) chunk_size_);
    if (num_chunks_ * chunk_size_ > num_nodes) {
        last_chunk_size_ = num_nodes - (num_chunks_ - 1) * chunk_size_;
    } else {
        last_chunk_size_ = chunk_size_;
    }

    y_ = std::vector<Matrix<float>>(num_chunks_);
    self_gradients_ = std::vector<Matrix<float>>(num_chunks_);
    neighbourhood_gradients_ = std::vector<Matrix<float>>(num_chunks_);
    long current_chunk_size = chunk_size_;
    for (int i = 0; i < num_chunks_; ++i) {
        if (i == (num_chunks_ - 1)) {
            current_chunk_size = last_chunk_size_;
        }
        y_.at(i).set(current_chunk_size, num_out_features, false);
        self_gradients_.at(i).set(current_chunk_size, num_in_features, false);
        neighbourhood_gradients_.at(i).set(current_chunk_size, num_in_features, false);
    }

    input_gradients_.self_gradients = &self_gradients_;
    input_gradients_.neighbourhood_gradients = &neighbourhood_gradients_;

    std::vector<Matrix<float> *> self_parameter_gradients = linear_self_.get_gradients();
    std::vector<Matrix<float> *> neigh_parameter_gradients = linear_neigh_.get_gradients();
    self_weight_sum_.set(self_parameter_gradients[0]->num_rows_, self_parameter_gradients[0]->num_columns_,
                         self_parameter_gradients[0]->is_row_major_);
    self_bias_sum_.set(self_parameter_gradients[1]->num_rows_, self_parameter_gradients[1]->num_columns_,
                       self_parameter_gradients[1]->is_row_major_);
    neigh_weight_sum_.set(neigh_parameter_gradients[0]->num_rows_, neigh_parameter_gradients[0]->num_columns_,
                          neigh_parameter_gradients[0]->is_row_major_);
    neigh_bias_sum_.set(neigh_parameter_gradients[1]->num_rows_, neigh_parameter_gradients[1]->num_columns_,
                        neigh_parameter_gradients[1]->is_row_major_);
}

std::vector<Matrix<float>> *SageLinearChunked::forward(std::vector<Matrix<float>> *features, std::vector<Matrix<float>> *aggr) {
    if (features->size() != aggr->size()) {
        throw "Features and aggregated features have a different number of chunks";
    }
    for (int i = 0; i < features->size(); ++i) {
        to_column_major_inplace(&features->at(i));
        to_column_major_inplace(&aggr->at(i));
    }

    Matrix<float> y_chunk;
    Matrix<float> *self_y;
    Matrix<float> *neigh_y;
    for (int i = 0; i < num_chunks_; ++i) {
        self_y = linear_self_.forward(&features->at(i));
        neigh_y = linear_neigh_.forward(&aggr->at(i));

        mat_mat_add(cuda_helper_, self_y, neigh_y, &y_.at(i));
    }

    features_ = features;
    aggregated_features_ = aggr;

    return &y_;
}

SageLinearGradientsChunked *SageLinearChunked::backward(std::vector<Matrix<float>> *incoming_gradients) {
    if (y_.size() != incoming_gradients->size()) {
        throw "Output and incoming gradients have a different number of chunks";
    }
    for (int i = 0; i < incoming_gradients->size(); ++i) {
        to_column_major_inplace(&incoming_gradients->at(i));
    }

    neigh_weight_sum_.set_values(0.0);
    self_bias_sum_.set_values(0.0);
    self_weight_sum_.set_values(0.0);
    neigh_bias_sum_.set_values(0.0);

    Matrix<float> *self_gradients;
    Matrix<float> *neigh_gradients;
    for (int i = 0; i < num_chunks_; ++i) {
        linear_self_.backward(&incoming_gradients->at(i), &features_->at(i), &self_gradients_.at(i));
        linear_neigh_.backward(&incoming_gradients->at(i), &aggregated_features_->at(i), &neighbourhood_gradients_.at(i));

        std::vector<Matrix<float> *> self_parameter_gradients = linear_self_.get_gradients();
        std::vector<Matrix<float> *> neigh_parameter_gradients = linear_neigh_.get_gradients();

        mat_mat_add(cuda_helper_, self_parameter_gradients[0], &self_weight_sum_, &self_weight_sum_);
        mat_mat_add(cuda_helper_, self_parameter_gradients[1], &self_bias_sum_, &self_bias_sum_);
        mat_mat_add(cuda_helper_, neigh_parameter_gradients[0], &neigh_weight_sum_, &neigh_weight_sum_);
        mat_mat_add(cuda_helper_, neigh_parameter_gradients[1], &neigh_bias_sum_, &neigh_bias_sum_);
    }

    linear_self_.set_gradients(&self_weight_sum_, &self_bias_sum_);
    linear_neigh_.set_gradients(&neigh_weight_sum_, &neigh_bias_sum_);

    return &input_gradients_;
}
