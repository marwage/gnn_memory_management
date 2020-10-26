// Copyright 2020 Marcel Wagenländer

#include "sage_linear.hpp"

#include <cmath>
#include <cstring>


SageLinear::SageLinear() {}

SageLinear::SageLinear(int in_features, int out_features, CudaHelper *helper) {
    cuda_helper_ = helper;

    num_in_features_ = in_features;
    num_out_features_ = out_features;
    linear_self_ = Linear(num_in_features_, num_out_features_, cuda_helper_);
    linear_neigh_ = Linear(num_in_features_, num_out_features_, cuda_helper_);
}

matrix<float> *SageLinear::get_parameters() {
    matrix<float> *self_params = linear_self_.get_parameters();
    matrix<float> *neigh_params = linear_neigh_.get_parameters();
    matrix<float> *params = (matrix<float> *) malloc(4 * sizeof(matrix<float>));
    params[0] = self_params[0];
    params[1] = self_params[1];
    params[2] = neigh_params[0];
    params[3] = neigh_params[1];

    return params;
}

matrix<float> *SageLinear::get_gradients() {
    matrix<float> *self_grads = linear_self_.get_gradients();
    matrix<float> *neigh_grads = linear_neigh_.get_gradients();
    matrix<float> *grads = (matrix<float> *) malloc(4 * sizeof(matrix<float>));
    grads[0] = self_grads[0];
    grads[1] = self_grads[1];
    grads[2] = neigh_grads[0];
    grads[3] = neigh_grads[1];

    return grads;
}

matrix<float> SageLinear::forward(matrix<float> features,
                                  matrix<float> aggr) {
    matrix<float> self_result = linear_self_.forward(features);
    matrix<float> neigh_result = linear_neigh_.forward(aggr);

    float *d_self;
    check_cuda(cudaMalloc((void **) &d_self,
                          self_result.rows * self_result.columns * sizeof(float)));
    check_cuda(cudaMemcpy(d_self, self_result.values,
                          self_result.rows * self_result.columns * sizeof(float),
                          cudaMemcpyHostToDevice));

    float *d_neigh;
    check_cuda(cudaMalloc((void **) &d_neigh,
                          neigh_result.rows * neigh_result.columns * sizeof(float)));
    check_cuda(cudaMemcpy(d_neigh, neigh_result.values,
                          neigh_result.rows * neigh_result.columns * sizeof(float),
                          cudaMemcpyHostToDevice));

    float alpha = 1.0;
    check_cublas(cublasSaxpy(cuda_helper_->cublas_handle,
                             self_result.rows * self_result.columns, &alpha,
                             d_neigh, 1,
                             d_self, 1));

    check_cuda(cudaMemcpy(self_result.values, d_self,
                          self_result.rows * self_result.columns * sizeof(float),
                          cudaMemcpyDeviceToHost));

    check_cuda(cudaFree(d_self));
    check_cuda(cudaFree(d_neigh));

    return self_result;
}

SageLinear::SageLinearGradients SageLinear::backward(matrix<float> in_gradients) {
    SageLinearGradients grads;

    grads.self_grads = linear_self_.backward(in_gradients);
    grads.neigh_grads = linear_neigh_.backward(in_gradients);

    return grads;
}

void SageLinear::update_weights(matrix<float> *gradients) {
    linear_self_.update_weights(gradients);
    linear_neigh_.update_weights(&gradients[2]);
}

SageLinearChunked::SageLinearChunked(CudaHelper *helper, int num_in_features, int num_out_features, int chunk_size) {
    chunk_size_ = chunk_size;
    sage_linear_layer_ = SageLinear(num_in_features, num_out_features, helper);
    num_out_features_ = num_out_features;
}

matrix<float> SageLinearChunked::forward(matrix<float> features, matrix<float> aggr){
    num_chunks_ = ceil((float) features.rows / (float) chunk_size_);

    if (num_chunks_ * chunk_size_ > features.rows) {
        last_chunk_size_ = features.rows - (num_chunks_ - 1) * chunk_size_;
    } else {
        last_chunk_size_ = chunk_size_;
    }

    to_row_major(&features);
    to_row_major(&aggr);

    matrix<float> Y;
    Y.rows = features.rows;
    Y.columns = num_out_features_;
    Y.values = reinterpret_cast<float *>(malloc(Y.rows * Y.columns * sizeof(float)));
    matrix<float> features_chunk;
    matrix<float> aggr_chunk;
    matrix<float> Y_chunk;

    for (int i = 0; i < num_chunks_; ++i) {
        if (i == (num_chunks_ - 1)) {
            features_chunk.rows = last_chunk_size_;
            aggr_chunk.rows = last_chunk_size_;
        } else {
            features_chunk.rows = chunk_size_;
            aggr_chunk.rows = chunk_size_;
        }
        features_chunk.columns = features.columns;
        features_chunk.values = &features.values[i * chunk_size_];
        to_column_major(&features_chunk);
        aggr_chunk.columns = aggr.columns;
        aggr_chunk.values = &aggr.values[i * chunk_size_];
        to_column_major(&aggr_chunk);

        Y_chunk = sage_linear_layer_.forward(features_chunk, aggr_chunk);
        to_row_major(&Y_chunk);

        std::memcpy(&Y.values[i * chunk_size_], Y_chunk.values, Y_chunk.rows * Y_chunk.columns * sizeof(float));
    }

    to_column_major(&Y);

    return Y;
}

SageLinear::SageLinearGradients SageLinearChunked::backward(matrix<float> in_gradients) {
    SageLinear::SageLinearGradients grads;
    grads.self_grads = in_gradients;
    grads.neigh_grads = in_gradients;
    return grads;
}

matrix<float> *SageLinearChunked::get_parameters() {
    return sage_linear_layer_.get_parameters();
}

matrix<float> *SageLinearChunked::get_gradients() {
    return sage_linear_layer_.get_gradients();
}

void SageLinearChunked::update_weights(matrix<float> *gradients) {
    sage_linear_layer_.update_weights(gradients);
}
