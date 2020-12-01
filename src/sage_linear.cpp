// Copyright 2020 Marcel Wagenländer

#include "sage_linear.hpp"

#include <cmath>
#include <string>


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
}

Matrix<float> **SageLinear::get_parameters() {
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
void SageLinear::set_parameters(Matrix<float> **parameters) {
    linear_self_.set_parameters(parameters);
    linear_neigh_.set_parameters(&parameters[2]);
}

Matrix<float> **SageLinear::get_gradients() {
    Matrix<float> **self_grads = linear_self_.get_gradients();
    Matrix<float> **neigh_grads = linear_neigh_.get_gradients();
    Matrix<float> **grads = new Matrix<float> *[4];
    grads[0] = self_grads[0];
    grads[1] = self_grads[1];
    grads[2] = neigh_grads[0];
    grads[3] = neigh_grads[1];

    return grads;
}

void SageLinear::set_gradients(Matrix<float> **grads) {
    linear_self_.set_gradients(grads);
    linear_neigh_.set_gradients(&grads[2]);
}

Matrix<float> *SageLinear::forward(Matrix<float> *features, Matrix<float> *aggr) {
    Matrix<float> *self_result = linear_self_.forward(features);
    Matrix<float> *neigh_result = linear_neigh_.forward(aggr);

    float *d_self;
    check_cuda(cudaMalloc((void **) &d_self,
                          self_result->num_rows_ * self_result->num_columns_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_self, self_result->values_,
                          self_result->num_rows_ * self_result->num_columns_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    float *d_neigh;
    check_cuda(cudaMalloc((void **) &d_neigh,
                          neigh_result->num_rows_ * neigh_result->num_columns_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_neigh, neigh_result->values_,
                          neigh_result->num_rows_ * neigh_result->num_columns_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    float alpha = 1.0;
    check_cublas(cublasSaxpy(cuda_helper_->cublas_handle,
                             self_result->num_rows_ * self_result->num_columns_, &alpha,
                             d_neigh, 1,
                             d_self, 1));

    check_cuda(cudaMemcpy(self_result->values_, d_self,
                          self_result->num_rows_ * self_result->num_columns_ * sizeof(float),
                          cudaMemcpyDeviceToHost));

    check_cuda(cudaFree(d_self));
    check_cuda(cudaFree(d_neigh));

    return self_result;
}

SageLinearGradients *SageLinear::backward(Matrix<float> *in_gradients) {
    input_gradients_.self_grads = linear_self_.backward(in_gradients);
    input_gradients_.neigh_grads = linear_neigh_.backward(in_gradients);

    return &input_gradients_;
}

void SageLinear::update_weights(Matrix<float> *gradients) {
    linear_self_.update_weights(gradients);
    linear_neigh_.update_weights(&gradients[2]);
}

SageLinearChunked::SageLinearChunked(CudaHelper *helper, long num_in_features, long num_out_features, long chunk_size, long num_nodes) {
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

    sage_linear_layers_ = std::vector<SageLinear>(num_chunks_);
    features_chunks_ = std::vector<Matrix<float>>(num_chunks_);
    aggr_chunks_ = std::vector<Matrix<float>>(num_chunks_);
    in_gradients_chunks_ = std::vector<Matrix<float>>(num_chunks_);
    long current_chunk_size = chunk_size_;
    for (int i = 0; i < num_chunks_; ++i) {
        if (i == (num_chunks_ - 1)) {
            current_chunk_size = last_chunk_size_;
        }
        sage_linear_layers_[i].set(cuda_helper_, num_in_features_, num_out_features_, current_chunk_size);
        features_chunks_[i].set(current_chunk_size, num_in_features, true);
        aggr_chunks_[i].set(current_chunk_size, num_in_features, true);
        in_gradients_chunks_[i].set(current_chunk_size, num_out_features, true);
    }

    if (num_chunks_ > 1) {
        Matrix<float> **params = sage_linear_layers_[0].get_parameters();
        for (int i = 1; i < num_chunks_; ++i) {
            sage_linear_layers_[i].set_parameters(params);
        }
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

    Matrix<float> *y_chunk;
    for (int i = 0; i < num_chunks_; ++i) {
        features_chunks_[i].is_row_major_ = features->is_row_major_;
        aggr_chunks_[i].is_row_major_ = aggr->is_row_major_;
        std::memcpy(features_chunks_[i].values_, &features->values_[i * chunk_size_ * features->num_columns_],
                    features_chunks_[i].num_rows_ * features_chunks_[i].num_columns_ * sizeof(float));
        std::memcpy(aggr_chunks_[i].values_, &aggr->values_[i * chunk_size_ * aggr->num_columns_],
                    aggr_chunks_[i].num_rows_ * aggr_chunks_[i].num_columns_ * sizeof(float));

        y_chunk = sage_linear_layers_[i].forward(&features_chunks_[i], &aggr_chunks_[i]);

        to_row_major_inplace(y_chunk);
        std::memcpy(&y_.values_[i * chunk_size_ * y_chunk->num_columns_], y_chunk->values_,
                    y_chunk->num_rows_ * y_chunk->num_columns_ * sizeof(float));
    }

    y_.is_row_major_ = true;

    return &y_;
}

SageLinearGradients *SageLinearChunked::backward(Matrix<float> *in_gradients) {
    to_row_major_inplace(in_gradients);

    SageLinearGradients *gradients_chunk;
    for (int i = 0; i < num_chunks_; ++i) {
        in_gradients_chunks_[i].is_row_major_ = in_gradients->is_row_major_;
        std::memcpy(in_gradients_chunks_[i].values_, &in_gradients->values_[i * chunk_size_ * in_gradients->num_columns_],
                    in_gradients_chunks_[i].num_rows_ * in_gradients_chunks_[i].num_columns_ * sizeof(float));

        gradients_chunk = sage_linear_layers_[i].backward(&in_gradients_chunks_[i]);

        to_row_major_inplace(gradients_chunk->self_grads);
        to_row_major_inplace(gradients_chunk->neigh_grads);
        std::memcpy(&input_gradients_.self_grads->values_[i * chunk_size_ * num_in_features_],
                    gradients_chunk->self_grads->values_,
                    gradients_chunk->self_grads->num_rows_ * gradients_chunk->self_grads->num_columns_ * sizeof(float));
        std::memcpy(&input_gradients_.neigh_grads->values_[i * chunk_size_ * num_in_features_],
                    gradients_chunk->neigh_grads->values_,
                    gradients_chunk->neigh_grads->num_rows_ * gradients_chunk->neigh_grads->num_columns_ * sizeof(float));
    }

    // add gradients of all layers
    if (num_chunks_ > 1) {
        Matrix<float> **gradients = sage_linear_layers_[0].get_gradients();
        int num_parameters = 4;
        float alpha = 1;
        float *d_sum;
        float *d_gradient;
        for (int j = 0; j < num_parameters; ++j) {
            check_cuda(cudaMalloc((void **) &d_sum,
                                  gradients[j]->num_rows_ * gradients[j]->num_columns_ * sizeof(float)));
            check_cuda(cudaMemcpy(d_sum, gradients[j]->values_,
                                  gradients[j]->num_rows_ * gradients[j]->num_columns_ * sizeof(float),
                                  cudaMemcpyHostToDevice));
            check_cuda(cudaMalloc((void **) &d_gradient,
                                  gradients[j]->num_rows_ * gradients[j]->num_columns_ * sizeof(float)));

            for (int i = 1; i < num_chunks_; ++i) {
                Matrix<float> **gradients_i = sage_linear_layers_[i].get_gradients();
                check_cuda(cudaMemcpy(d_gradient, gradients_i[j]->values_,
                                      gradients_i[j]->num_rows_ * gradients_i[j]->num_columns_ * sizeof(float),
                                      cudaMemcpyHostToDevice));
                check_cublas(cublasSaxpy(cuda_helper_->cublas_handle, gradients[j]->num_rows_ * gradients[j]->num_columns_,
                                         &alpha, d_gradient, 1, d_sum, 1));
            }

            check_cuda(cudaMemcpy(gradients[j]->values_, d_sum,
                                  gradients[j]->num_rows_ * gradients[j]->num_columns_ * sizeof(float),
                                  cudaMemcpyDeviceToHost));

            cudaFree(d_sum);
            cudaFree(d_gradient);
        }

        for (int i = 1; i < num_chunks_; ++i) {
            sage_linear_layers_[i].set_gradients(sage_linear_layers_[0].get_gradients());
        }
    }

    input_gradients_.neigh_grads->is_row_major_ = true;
    input_gradients_.neigh_grads->is_row_major_ = true;

    return &input_gradients_;
}

Matrix<float> **SageLinearChunked::get_parameters() {
    return sage_linear_layers_[0].get_parameters();
}

void SageLinearChunked::set_parameters(Matrix<float> **parameters) {
    for (int i = 0; i < num_chunks_; ++i) {
        sage_linear_layers_[i].set_parameters(parameters);
    }
}

Matrix<float> **SageLinearChunked::get_gradients() {
    return sage_linear_layers_[0].get_gradients();
}

void SageLinearChunked::update_weights(Matrix<float> *gradients) {
    for (int i = 0; i < num_chunks_; ++i) {
        sage_linear_layers_[i].update_weights(gradients);
    }
}

std::vector<SageLinear> *SageLinearChunked::get_layers() {
    return &sage_linear_layers_;
}
