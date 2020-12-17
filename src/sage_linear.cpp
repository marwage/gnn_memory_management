// Copyright 2020 Marcel Wagenländer

#include "sage_linear.hpp"
#include "dense_computation.hpp"

#include <cmath>


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

std::vector<Matrix<float> *> SageLinear::get_parameters() {
    std::vector<Matrix<float> *> self_params = linear_self_.get_parameters();
    std::vector<Matrix<float> *> neigh_params = linear_neigh_.get_parameters();
    std::vector<Matrix<float> *> parameters(4);
    parameters[0] = self_params[0];
    parameters[1] = self_params[1];
    parameters[2] = neigh_params[0];
    parameters[3] = neigh_params[1];

    return parameters;
}

std::vector<Matrix<float> *> SageLinear::get_gradients() {
    std::vector<Matrix<float> *> self_grads = linear_self_.get_gradients();
    std::vector<Matrix<float> *> neigh_grads = linear_neigh_.get_gradients();
    std::vector<Matrix<float> *> gradients(4);
    gradients[0] = self_grads[0];
    gradients[1] = self_grads[1];
    gradients[2] = neigh_grads[0];
    gradients[3] = neigh_grads[1];

    return gradients;
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

// CHUNKED --- CHUNKED --- CHUNKED

SageLinearChunked::SageLinearChunked() {}

SageLinearChunked::SageLinearChunked(CudaHelper *helper, long num_in_features, long num_out_features, long chunk_size, long num_nodes) {
    set(helper, num_in_features, num_out_features, chunk_size, num_nodes);
}

void SageLinearChunked::set(CudaHelper *helper, long num_in_features, long num_out_features, long chunk_size, long num_nodes) {
    cuda_helper_ = helper;
    chunk_size_ = chunk_size;
    num_in_features_ = num_in_features;
    num_out_features_ = num_out_features;

    num_chunks_ = ceil((float) num_nodes / (float) chunk_size_);
    if (num_chunks_ * chunk_size_ > num_nodes) {
        last_chunk_size_ = num_nodes - (num_chunks_ - 1) * chunk_size_;
    } else {
        last_chunk_size_ = chunk_size_;
    }

    linear_self_.set(cuda_helper_, chunk_size, num_nodes, num_in_features_, num_out_features_);
    linear_neigh_.set(cuda_helper_, chunk_size, num_nodes, num_in_features_, num_out_features_);
    add_.set(cuda_helper_, chunk_size, num_nodes, num_out_features);
}

std::vector<Matrix<float> *> SageLinearChunked::get_parameters() {
    std::vector<Matrix<float> *> self_params = linear_self_.get_parameters();
    std::vector<Matrix<float> *> neigh_params = linear_neigh_.get_parameters();
    std::vector<Matrix<float> *> parameters(4);
    parameters[0] = self_params[0];
    parameters[1] = self_params[1];
    parameters[2] = neigh_params[0];
    parameters[3] = neigh_params[1];

    return parameters;
}

std::vector<Matrix<float> *> SageLinearChunked::get_gradients() {
    std::vector<Matrix<float> *> self_grads = linear_self_.get_gradients();
    std::vector<Matrix<float> *> neigh_grads = linear_neigh_.get_gradients();
    std::vector<Matrix<float> *> gradients(4);
    gradients[0] = self_grads[0];
    gradients[1] = self_grads[1];
    gradients[2] = neigh_grads[0];
    gradients[3] = neigh_grads[1];

    return gradients;
}

std::vector<Matrix<float>> *SageLinearChunked::forward(std::vector<Matrix<float>> *features, std::vector<Matrix<float>> *aggr) {
    if (features->size() != aggr->size()) {
        throw "Features and aggregated features have a different number of chunks";
    }
    for (int i = 0; i < num_chunks_; ++i) {
        to_column_major_inplace(&features->at(i));
        to_column_major_inplace(&aggr->at(i));
    }

    std::vector<Matrix<float>> *y_self = linear_self_.forward(features);
    std::vector<Matrix<float>> *y_neigh = linear_neigh_.forward(aggr);
    y_ = add_.forward(y_self, y_neigh);

    return y_;
}

SageLinearGradientsChunked *SageLinearChunked::backward(std::vector<Matrix<float>> *incoming_gradients) {
    if (y_->size() != incoming_gradients->size()) {
        throw "Output and incoming gradients have a different number of chunks";
    }
    for (int i = 0; i < num_chunks_; ++i) {
        to_column_major_inplace(&incoming_gradients->at(i));
    }

    input_gradients_.self_gradients = linear_self_.backward(incoming_gradients);
    input_gradients_.neighbourhood_gradients = linear_neigh_.backward(incoming_gradients);

    return &input_gradients_;
}

// PIPELINED --- PIPELINED --- PIPELINED

SageLinearPipelined::SageLinearPipelined() {}

SageLinearPipelined::SageLinearPipelined(CudaHelper *helper, long num_in_features, long num_out_features, long chunk_size, long num_nodes) {
    set(helper, num_in_features, num_out_features, chunk_size, num_nodes);
}

void SageLinearPipelined::set(CudaHelper *helper, long num_in_features, long num_out_features, long chunk_size, long num_nodes) {
    SageLinearChunked::set(helper, num_in_features, num_out_features, chunk_size, num_nodes);

    num_steps_ = 3;

    linear_self_.set(cuda_helper_, chunk_size, num_nodes, num_in_features, num_out_features);
    linear_neigh_.set(cuda_helper_, chunk_size, num_nodes, num_in_features, num_out_features);
    ;
}

void SageLinearPipelined::forward_in(long chunk, long buffer) {
}

void SageLinearPipelined::forward_out(long chunk, long buffer) {
}

void SageLinearPipelined::forward_compute(long chunk, long buffer) {
}

std::vector<Matrix<float>> *SageLinearPipelined::forward(std::vector<Matrix<float>> *features, std::vector<Matrix<float>> *aggr) {
    std::vector<Matrix<float>> *y_self = linear_self_.forward(features);

    std::vector<Matrix<float>> *y_neigh = linear_neigh_.forward(aggr);


    return y_;
}

void SageLinearPipelined::backward_in(long chunk, long buffer) {
}

void SageLinearPipelined::backward_out(long chunk, long buffer) {
}

void SageLinearPipelined::backward_compute(long chunk, long buffer) {
}

SageLinearGradientsChunked *SageLinearPipelined::backward(std::vector<Matrix<float>> *incoming_gradients) {

    return &input_gradients_;
}
