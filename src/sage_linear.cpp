// Copyright 2020 Marcel Wagenländer

#include "sage_linear.hpp"
#include "dense_computation.hpp"

#include <cmath>
#include <string>


Matrix<float> **SageLinearParent::get_parameters() {
    Matrix<float> **self_params = linear_self_.get_parameters();
    Matrix<float> **neigh_params = linear_neigh_.get_parameters();
    Matrix<float> **params = new Matrix<float> *[4];
    params[0] = self_params[0];
    params[1] = self_params[1];
    params[2] = neigh_params[0];
    params[3] = neigh_params[1];

    return params;
}

// assume number of parameters is 4
void SageLinearParent::set_parameters(Matrix<float> **parameters) {
    linear_self_.set_parameters(parameters);
    linear_neigh_.set_parameters(&parameters[2]);
}

Matrix<float> **SageLinearParent::get_gradients() {
    Matrix<float> **self_grads = linear_self_.get_gradients();
    Matrix<float> **neigh_grads = linear_neigh_.get_gradients();
    Matrix<float> **grads = new Matrix<float> *[4];
    grads[0] = self_grads[0];
    grads[1] = self_grads[1];
    grads[2] = neigh_grads[0];
    grads[3] = neigh_grads[1];

    return grads;
}

void SageLinearParent::set_gradients(Matrix<float> **grads) {
    linear_self_.set_gradients(grads);
    linear_neigh_.set_gradients(&grads[2]);
}

void SageLinearParent::update_weights(Matrix<float> *gradients) {
    linear_self_.update_weights(gradients);
    linear_neigh_.update_weights(&gradients[2]);
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
    input_gradients_.self_grads = linear_self_.backward(in_gradients);
    input_gradients_.neigh_grads = linear_neigh_.backward(in_gradients);

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

    features_chunks_ = std::vector<Matrix<float>>(num_chunks_);
    aggr_chunks_ = std::vector<Matrix<float>>(num_chunks_);
    in_gradients_chunks_ = std::vector<Matrix<float>>(num_chunks_);
    long current_chunk_size = chunk_size_;
    for (int i = 0; i < num_chunks_; ++i) {
        if (i == (num_chunks_ - 1)) {
            current_chunk_size = last_chunk_size_;
        }
        features_chunks_[i].set(current_chunk_size, num_in_features, true);
        aggr_chunks_[i].set(current_chunk_size, num_in_features, true);
        in_gradients_chunks_[i].set(current_chunk_size, num_out_features, true);
    }

    y_.set(num_nodes, num_out_features, true);

    self_gradients_.set(num_nodes, num_in_features, true);
    neighbourhood_gradients_.set(num_nodes, num_in_features, true);
    input_gradients_.self_grads = &self_gradients_;
    input_gradients_.neigh_grads = &neighbourhood_gradients_;
}

Matrix<float> *SageLinearChunked::forward(Matrix<float> *features, Matrix<float> *aggr) {
    to_row_major_inplace(features);
    to_row_major_inplace(aggr);

    Matrix<float> features_chunk_row;
    Matrix<float> aggr_chunk_row;
    for (int i = 0; i < num_chunks_; ++i) {
        features_chunk_row.set(features_chunks_[i].num_rows_, features_chunks_[i].num_columns_,
                        &features->values_[i * chunk_size_ * features->num_columns_],
                               features_chunks_[i].is_row_major_, false);
        to_column_major(&features_chunks_[i], &features_chunk_row);

        aggr_chunk_row.set(aggr_chunks_[i].num_rows_, aggr_chunks_[i].num_columns_,
                               &aggr->values_[i * chunk_size_ * aggr->num_columns_],
                           aggr_chunks_[i].is_row_major_, false);
        to_column_major(&aggr_chunks_[i], &aggr_chunk_row);
    }

    Matrix<float> y_chunk;
    Matrix<float> *self_y;
    Matrix<float> *neigh_y;
    for (int i = 0; i < num_chunks_; ++i) {
        self_y = linear_self_.forward(&features_chunks_[i]);
        neigh_y = linear_neigh_.forward(&aggr_chunks_[i]);

        if (i == num_chunks_ - 1) {
            y_chunk.set(last_chunk_size_, num_out_features_,
                        &y_.values_[i * chunk_size_ * y_.num_columns_],
                        true, false);
        } else {
            y_chunk.set(chunk_size_, num_out_features_,
                        &y_.values_[i * chunk_size_ * y_.num_columns_],
                        true, false);
        }

        mat_mat_add(cuda_helper_, self_y, neigh_y, &y_chunk);

        to_row_major_inplace(&y_chunk);
    }

    y_.is_row_major_ = true;

    return &y_;
}

SageLinearGradients *SageLinearChunked::backward(Matrix<float> *in_gradients) {
    to_row_major_inplace(in_gradients);

    Matrix<float> in_gradients_row;
    for (int i = 0; i < num_chunks_; ++i) {
        in_gradients_row.set(in_gradients_chunks_[i].num_rows_, in_gradients_chunks_[i].num_columns_,
                               &in_gradients->values_[i * chunk_size_ * in_gradients->num_columns_],
                             in_gradients_chunks_[i].is_row_major_, false);
        to_column_major(&in_gradients_chunks_[i], &in_gradients_row);
    }

    Matrix<float> **self_parameter_gradients = linear_self_.get_gradients();
    Matrix<float> **neigh_parameter_gradients = linear_neigh_.get_gradients();
    Matrix<float> self_weight_sum(self_parameter_gradients[0]->num_rows_, self_parameter_gradients[0]->num_columns_,
                                  self_parameter_gradients[0]->is_row_major_);
    Matrix<float> self_bias_sum(self_parameter_gradients[1]->num_rows_, self_parameter_gradients[1]->num_columns_,
                                self_parameter_gradients[1]->is_row_major_);
    Matrix<float> neigh_weight_sum(neigh_parameter_gradients[0]->num_rows_, neigh_parameter_gradients[0]->num_columns_,
                                   neigh_parameter_gradients[0]->is_row_major_);
    Matrix<float> neigh_bias_sum(neigh_parameter_gradients[1]->num_rows_, neigh_parameter_gradients[1]->num_columns_,
                                 neigh_parameter_gradients[1]->is_row_major_);

    Matrix<float> *self_gradients;
    Matrix<float> *neigh_gradients;
    for (int i = 0; i < num_chunks_; ++i) {
        self_gradients = linear_self_.backward(&in_gradients_chunks_[i]), &features_chunks_[i];
        neigh_gradients = linear_neigh_.backward(&in_gradients_chunks_[i], &aggr_chunks_[i]);

        to_row_major_inplace(self_gradients);
        to_row_major_inplace(neigh_gradients);

        std::memcpy(&input_gradients_.self_grads->values_[i * chunk_size_ * num_out_features_],
                    self_gradients->values_,
                    self_gradients->size_ * sizeof(float));
        std::memcpy(&input_gradients_.neigh_grads->values_[i * chunk_size_ * num_out_features_],
                    neigh_gradients->values_,
                    neigh_gradients->size_ * sizeof(float));

        Matrix<float> **self_parameter_gradients = linear_self_.get_gradients();
        Matrix<float> **neigh_parameter_gradients = linear_neigh_.get_gradients();

        mat_mat_add(cuda_helper_, self_parameter_gradients[0], &self_weight_sum, &self_weight_sum);
        mat_mat_add(cuda_helper_, self_parameter_gradients[1], &self_bias_sum, &self_bias_sum);
        mat_mat_add(cuda_helper_, neigh_parameter_gradients[0], &neigh_weight_sum, &neigh_weight_sum);
        mat_mat_add(cuda_helper_, neigh_parameter_gradients[1], &neigh_bias_sum, &neigh_bias_sum);
    }

    linear_self_.set_parameters(&self_weight_sum, &self_bias_sum);
    linear_neigh_.set_parameters(&neigh_weight_sum, &neigh_bias_sum);

    input_gradients_.neigh_grads->is_row_major_ = true;
    input_gradients_.neigh_grads->is_row_major_ = true;

    return &input_gradients_;
}
