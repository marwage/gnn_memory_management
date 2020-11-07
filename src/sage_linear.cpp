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

// assume number of parameters is 4
void SageLinear::set_parameters(matrix<float> *parameters) {
    linear_self_.set_parameters(parameters);
    linear_neigh_.set_parameters(&parameters[2]);
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

void SageLinear::set_gradients(matrix<float> *grads) {
    linear_self_.set_gradients(grads);
    linear_neigh_.set_gradients(&grads[2]);
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

SageLinearChunked::SageLinearChunked(CudaHelper *helper, int num_in_features, int num_out_features, int chunk_size, int num_nodes) {
    cuda_helper_ = helper;
    chunk_size_ = chunk_size;
    num_in_features_ = num_in_features;
    num_out_features_ = num_out_features;
    input_shape_ = std::vector<int>(2);

    num_chunks_ = ceil((float) num_nodes / (float) chunk_size_);
    if (num_chunks_ * chunk_size_ > num_nodes) {
        last_chunk_size_ = num_nodes - (num_chunks_ - 1) * chunk_size_;
    } else {
        last_chunk_size_ = chunk_size_;
    }

    sage_linear_layers_ = std::vector<SageLinear>(num_chunks_);
    for (int i = 0; i < num_chunks_; ++i) {
        sage_linear_layers_[i] = SageLinear(num_in_features_, num_out_features_, cuda_helper_);
    }
    if (num_chunks_ > 1) {
        for (int i = 1; i < num_chunks_; ++i) {
            sage_linear_layers_[i].set_parameters(sage_linear_layers_[0].get_parameters());
        }
    }
}

matrix<float> SageLinearChunked::forward(matrix<float> features, matrix<float> aggr){
    input_shape_ = {features.rows, features.columns};

    matrix<float> features_row = to_row_major(&features);
    matrix<float> aggr_row = to_row_major(&aggr);

    matrix<float> Y;
    Y.rows = features.rows;
    Y.columns = num_out_features_;
    Y.values = reinterpret_cast<float *>(malloc(Y.rows * Y.columns * sizeof(float)));
    matrix<float> features_chunk;
    matrix<float> aggr_chunk;
    matrix<float> Y_chunk;
    features_chunk.rows = chunk_size_;
    aggr_chunk.rows = chunk_size_;

    for (int i = 0; i < num_chunks_; ++i) {
        if (i == (num_chunks_ - 1)) {
            features_chunk.rows = last_chunk_size_;
            aggr_chunk.rows = last_chunk_size_;
        }
        features_chunk.columns = features_row.columns;
        features_chunk.values = &features_row.values[i * chunk_size_ * features_chunk.columns];
        to_column_major_inplace(&features_chunk);
        aggr_chunk.columns = aggr_row.columns;
        aggr_chunk.values = &aggr_row.values[i * chunk_size_ * aggr_row.columns];
        to_column_major_inplace(&aggr_chunk);

        Y_chunk = sage_linear_layers_[i].forward(features_chunk, aggr_chunk);
        to_row_major_inplace(&Y_chunk);

        std::memcpy(&Y.values[i * chunk_size_ * Y_chunk.columns], Y_chunk.values, Y_chunk.rows * Y_chunk.columns * sizeof(float));
    }

    to_column_major_inplace(&Y);

    return Y;
}

SageLinear::SageLinearGradients SageLinearChunked::backward(matrix<float> in_gradients) {
    matrix<float> in_gradients_row = to_row_major(&in_gradients);

    SageLinear::SageLinearGradients input_gradients;
    input_gradients.self_grads.rows = input_shape_[0];
    input_gradients.self_grads.columns = input_shape_[1];
    input_gradients.self_grads.values = reinterpret_cast<float *>(malloc(input_gradients.self_grads.rows * input_gradients.self_grads.columns * sizeof(float)));
    input_gradients.neigh_grads.rows = input_shape_[0];
    input_gradients.neigh_grads.columns = input_shape_[1];
    input_gradients.neigh_grads.values = reinterpret_cast<float *>(malloc(input_gradients.neigh_grads.rows * input_gradients.neigh_grads.columns * sizeof(float)));
    SageLinear::SageLinearGradients gradients_chunk;
    matrix<float> in_gradients_chunk;
    in_gradients_chunk.rows = chunk_size_;
    in_gradients_chunk.columns = in_gradients_row.columns;

    for (int i = 0; i < num_chunks_; ++i) {
        if (i == (num_chunks_ - 1)) {
            in_gradients_chunk.rows = last_chunk_size_;
        }
        in_gradients_chunk.values = &in_gradients_row.values[i * chunk_size_ * in_gradients_row.columns];
        to_column_major_inplace(&in_gradients_chunk);

        gradients_chunk = sage_linear_layers_[i].backward(in_gradients_chunk);
        to_row_major_inplace(&gradients_chunk.self_grads);
        to_row_major_inplace(&gradients_chunk.neigh_grads);

        std::memcpy(&input_gradients.self_grads.values[i * chunk_size_ * gradients_chunk.self_grads.columns], gradients_chunk.self_grads.values, gradients_chunk.self_grads.rows * gradients_chunk.self_grads.columns * sizeof(float));
        std::memcpy(&input_gradients.neigh_grads.values[i * chunk_size_ * gradients_chunk.neigh_grads.columns], gradients_chunk.neigh_grads.values, gradients_chunk.neigh_grads.rows * gradients_chunk.neigh_grads.columns * sizeof(float));
    }

    // add gradients of all layers
    if (num_chunks_ > 1) {
        matrix<float> *gradients = sage_linear_layers_[0].get_gradients();
        int num_parameters = 4;
        float alpha = 1;
        float *d_sum;
        float *d_gradient;
        for (int j = 0; j < num_parameters; ++j) {
            check_cuda(cudaMalloc((void **) &d_sum,
                                  gradients[j].rows * gradients[j].columns * sizeof(float)));
            check_cuda(cudaMemcpy(d_sum, gradients[j].values,
                                  gradients[j].rows * gradients[j].columns * sizeof(float),
                                  cudaMemcpyHostToDevice));
            check_cuda(cudaMalloc((void **) &d_gradient,
                                  gradients[j].rows * gradients[j].columns * sizeof(float)));

            for (int i = 1; i < num_chunks_; ++i) {
                matrix<float> *gradients_i = sage_linear_layers_[i].get_gradients();
                check_cuda(cudaMemcpy(d_gradient, gradients_i[j].values,
                                      gradients_i[j].rows * gradients_i[j].columns * sizeof(float),
                                      cudaMemcpyHostToDevice));
                check_cublas(cublasSaxpy(cuda_helper_->cublas_handle, gradients[j].rows * gradients[j].columns,
                                         &alpha, d_gradient, 1, d_sum, 1));
            }

            check_cuda(cudaMemcpy(gradients[j].values, d_sum,
                                  gradients[j].rows * gradients[j].columns * sizeof(float),
                                  cudaMemcpyDeviceToHost));

            cudaFree(d_sum);
            cudaFree(d_gradient);

        }

        for (int i = 1; i < num_chunks_; ++i) {
            sage_linear_layers_[i].set_gradients(sage_linear_layers_[0].get_gradients());
        }
    }

    to_column_major_inplace(&input_gradients.self_grads);
    to_column_major_inplace(&input_gradients.neigh_grads);

    return input_gradients;
}

matrix<float> *SageLinearChunked::get_parameters() {
    return sage_linear_layers_[0].get_parameters();
}

matrix<float> *SageLinearChunked::get_gradients() {
    return sage_linear_layers_[0].get_gradients();
}

void SageLinearChunked::update_weights(matrix<float> *gradients) {
    for (int i = 0; i < num_chunks_; ++i) {
        sage_linear_layers_[i].update_weights(gradients);
    }
}
