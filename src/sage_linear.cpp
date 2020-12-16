// Copyright 2020 Marcel Wagenländer

#include "sage_linear.hpp"
#include "dense_computation.hpp"

#include <cmath>


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

// CHUNKED --- CHUNKED --- CHUNKED

SageLinearChunked::SageLinearChunked() {}

SageLinearChunked::SageLinearChunked(CudaHelper *helper, long num_in_features, long num_out_features, long chunk_size, long num_nodes) {
    set(helper, num_in_features, num_out_features, chunk_size, num_nodes);
}

void SageLinearChunked::set(CudaHelper *helper, long num_in_features, long num_out_features, long chunk_size, long num_nodes){
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

    if (num_chunks_ == 1) {
        linear_self_.set(cuda_helper_, num_in_features_, num_out_features_, last_chunk_size_);
        linear_neigh_.set(cuda_helper_, num_in_features_, num_out_features_, last_chunk_size_);
    } else {
        linear_self_.set(cuda_helper_, num_in_features_, num_out_features_, chunk_size_);
        linear_neigh_.set(cuda_helper_, num_in_features_, num_out_features_, chunk_size_);
    }


    y_self_ = std::vector<Matrix<float>>(num_chunks_);
    y_neigh_ = std::vector<Matrix<float>>(num_chunks_);
    y_ = std::vector<Matrix<float>>(num_chunks_);
    self_gradients_ = std::vector<Matrix<float>>(num_chunks_);
    neighbourhood_gradients_ = std::vector<Matrix<float>>(num_chunks_);
    long current_chunk_size = chunk_size_;
    for (int i = 0; i < num_chunks_; ++i) {
        if (i == (num_chunks_ - 1)) {
            current_chunk_size = last_chunk_size_;
        }
        y_self_.at(i).set(current_chunk_size, num_out_features, false);
        y_neigh_.at(i).set(current_chunk_size, num_out_features, false);
        y_.at(i).set(current_chunk_size, num_out_features, false);
        self_gradients_.at(i).set(current_chunk_size, num_in_features, false);
        neighbourhood_gradients_.at(i).set(current_chunk_size, num_in_features, false);
    }

    input_gradients_.self_gradients = &self_gradients_;
    input_gradients_.neighbourhood_gradients = &neighbourhood_gradients_;

    std::vector<Matrix<float> *> self_parameter_gradients = linear_self_.get_gradients();
    std::vector<Matrix<float> *> neigh_parameter_gradients = linear_neigh_.get_gradients();
}

std::vector<Matrix<float>> *SageLinearChunked::forward(std::vector<Matrix<float>> *features, std::vector<Matrix<float>> *aggr) {
    features_ = features;
    aggregated_features_ = aggr;

    if (features->size() != aggr->size()) {
        throw "Features and aggregated features have a different number of chunks";
    }
    for (int i = 0; i < num_chunks_; ++i) {
        to_column_major_inplace(&features->at(i));
        to_column_major_inplace(&aggr->at(i));
    }

    linear_self_.forward_init();
    float *d_x;
    check_cuda(cudaMalloc(&d_x, features->at(0).size_ * sizeof(float)));
    float *d_y;

    for (int i = 0; i < num_chunks_; ++i) {
        // in
        check_cuda(cudaMemcpy(d_x, features->at(i).values_, features->at(i).size_ * sizeof(float),
                              cudaMemcpyHostToDevice));

        // compute
        d_y = linear_self_.forward_compute(d_x, features->at(i).num_rows_);

        // out
        check_cuda(cudaMemcpy(y_self_.at(i).values_, d_y, y_self_.at(i).size_ * sizeof(float),
                   cudaMemcpyDeviceToHost));
        y_self_.at(i).is_row_major_ = false;
    }
    linear_self_.forward_free();

    linear_neigh_.forward_init();
    for (int i = 0; i < num_chunks_; ++i) {
        // in
        check_cuda(cudaMemcpy(d_x, aggr->at(i).values_, aggr->at(i).size_ * sizeof(float),
                              cudaMemcpyHostToDevice));

        // compute
        d_y = linear_neigh_.forward_compute(d_x, aggr->at(i).num_rows_);

        // out
        check_cuda(cudaMemcpy(y_neigh_.at(i).values_, d_y, y_neigh_.at(i).size_ * sizeof(float),
                              cudaMemcpyDeviceToHost));
        y_neigh_.at(i).is_row_major_ = false;
    }
    check_cuda(cudaFree(d_x));
    linear_neigh_.forward_free();

    float *d_y_self;
    check_cuda(cudaMalloc(&d_y_self, y_self_.at(0).size_ * sizeof(float)));
    float *d_y_neigh;
    check_cuda(cudaMalloc(&d_y_neigh, y_neigh_.at(0).size_ * sizeof(float)));
    for (int i = 0; i < num_chunks_; ++i) {
        // in
        check_cuda(cudaMemcpy(d_y_self, y_self_.at(i).values_, y_self_.at(i).size_ * sizeof(float),
                              cudaMemcpyHostToDevice));
        check_cuda(cudaMemcpy(d_y_neigh, y_neigh_.at(i).values_, y_neigh_.at(i).size_ * sizeof(float),
                              cudaMemcpyHostToDevice));

        // compute
        mat_mat_add_cuda(cuda_helper_, d_y_self, d_y_neigh, y_self_.at(i).size_);

        // out
        check_cuda(cudaMemcpy(y_.at(i).values_, d_y_neigh, y_neigh_.at(i).size_ * sizeof(float),
                              cudaMemcpyDeviceToHost));
        y_.at(i).is_row_major_ = false;
    }
    check_cuda(cudaFree(d_y_self));
    check_cuda(cudaFree(d_y_neigh));

    return &y_;
}

SageLinearGradientsChunked *SageLinearChunked::backward(std::vector<Matrix<float>> *incoming_gradients) {
    if (y_.size() != incoming_gradients->size()) {
        throw "Output and incoming gradients have a different number of chunks";
    }
    for (int i = 0; i < num_chunks_; ++i) {
        to_column_major_inplace(&incoming_gradients->at(i));
    }

    float *d_weight_sum;
    check_cuda(cudaMalloc(&d_weight_sum, num_in_features_ * num_out_features_ * sizeof(float)));
    check_cuda((cudaMemset(d_weight_sum, 0, num_in_features_ * num_out_features_ * sizeof(float))));
    float *d_bias_sum;
    check_cuda(cudaMalloc(&d_bias_sum, num_out_features_ * sizeof(float)));
    check_cuda(cudaMemset(d_bias_sum, 0, num_out_features_ * sizeof(float)));

    float *d_dy;
    check_cuda(cudaMalloc(&d_dy, incoming_gradients->at(0).size_ * sizeof(float)));
    float *d_x;
    check_cuda((cudaMalloc(&d_x, features_->at(0).size_ * sizeof(float))));
    float *d_dx;
    std::vector<float *> gradients_cuda;

    linear_self_.backward_init();
    for (int i = 0; i < num_chunks_; ++i) {
        // in
        check_cuda(cudaMemcpy(d_dy, incoming_gradients->at(i).values_, incoming_gradients->at(i).size_ * sizeof(float),
                              cudaMemcpyHostToDevice));
        check_cuda(cudaMemcpy(d_x, features_->at(i).values_, features_->at(i).size_ * sizeof(float),
                              cudaMemcpyHostToDevice));

        // compute
        d_dx = linear_self_.backward_compute(d_dy, d_x);
        gradients_cuda = linear_self_.get_gradients_cuda();
        mat_mat_add_cuda(cuda_helper_, gradients_cuda.at(0), d_weight_sum, num_in_features_ * num_out_features_);
        mat_mat_add_cuda(cuda_helper_, gradients_cuda.at(1), d_bias_sum, num_out_features_);

        // out
        check_cuda(cudaMemcpy(input_gradients_.self_gradients->at(i).values_, d_dx, input_gradients_.self_gradients->at(i).size_ * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }
    linear_self_.backward_free();

    std::vector<Matrix<float> *> gradients = linear_self_.get_gradients();
    check_cuda(cudaMemcpy(gradients.at(0)->values_, d_weight_sum, num_in_features_ * num_out_features_ * sizeof(float),
                          cudaMemcpyDeviceToHost));
    check_cuda(cudaMemcpy(gradients.at(1)->values_, d_bias_sum, num_out_features_ * sizeof(float),
                          cudaMemcpyDeviceToHost));

    check_cuda((cudaMemset(d_weight_sum, 0, num_in_features_ * num_out_features_ * sizeof(float))));
    check_cuda(cudaMemset(d_bias_sum, 0, num_out_features_ * sizeof(float)));

    linear_neigh_.backward_init();
    for (int i = 0; i < num_chunks_; ++i) {
        // in
        check_cuda(cudaMemcpy(d_dy, incoming_gradients->at(i).values_, incoming_gradients->at(i).size_ * sizeof(float),
                              cudaMemcpyHostToDevice));
        check_cuda(cudaMemcpy(d_x, aggregated_features_->at(i).values_, aggregated_features_->at(i).size_ * sizeof(float),
                              cudaMemcpyHostToDevice));
        // backward neigh
        d_dx = linear_neigh_.backward_compute(d_dy, d_x);
        gradients_cuda = linear_neigh_.get_gradients_cuda();
        mat_mat_add_cuda(cuda_helper_, gradients_cuda.at(0), d_weight_sum, num_in_features_ * num_out_features_);
        mat_mat_add_cuda(cuda_helper_, gradients_cuda.at(1), d_bias_sum, num_out_features_);

        // out
        check_cuda(cudaMemcpy(input_gradients_.neighbourhood_gradients->at(i).values_, d_dx, input_gradients_.neighbourhood_gradients->at(i).size_ * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }
    linear_neigh_.backward_free();

    gradients = linear_neigh_.get_gradients();
    check_cuda(cudaMemcpy(gradients.at(0)->values_, d_weight_sum, num_in_features_ * num_out_features_ * sizeof(float),
                          cudaMemcpyDeviceToHost));
    check_cuda(cudaMemcpy(gradients.at(1)->values_, d_bias_sum, num_out_features_ * sizeof(float),
                          cudaMemcpyDeviceToHost));

    check_cuda(cudaFree(d_dy));
    check_cuda(cudaFree(d_x));
    check_cuda(cudaFree(d_weight_sum));
    check_cuda(cudaFree(d_bias_sum));

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

    d_x_ = std::vector<float *>(num_steps_);
    d_y_ = std::vector<float *>(num_steps_);
}

void SageLinearPipelined::forward_in(long chunk, long buffer) {
    check_cuda(cudaMemcpy(d_x_.at(buffer), features_->at(chunk).values_, features_->at(chunk).size_ * sizeof(float),
                          cudaMemcpyHostToDevice));
}

void SageLinearPipelined::forward_out(long chunk, long buffer) {
    check_cuda(cudaMemcpy(y_self_.at(chunk).values_, d_y_.at(buffer), y_self_.at(chunk).size_ * sizeof(float),
                          cudaMemcpyDeviceToHost));
    y_self_.at(chunk).is_row_major_ = false;
}

void SageLinearPipelined::forward_compute(long chunk, long buffer) {
    d_y_.at(buffer) = linear_neigh_.forward_compute(d_x_.at(buffer), aggregated_features_->at(chunk).num_rows_);
}

void SageLinearPipelined::forward_compute(long buffer) {
    // TESTING
}

std::vector<Matrix<float>> *SageLinearPipelined::forward(std::vector<Matrix<float>> *features, std::vector<Matrix<float>> *aggr) {
    features_ = features;
    aggregated_features_ = aggr;

    if (features->size() != aggr->size()) {
        throw "Features and aggregated features have a different number of chunks";
    }
    for (int i = 0; i < num_chunks_; ++i) {
        to_column_major_inplace(&features->at(i));
        to_column_major_inplace(&aggr->at(i));
    }

    linear_self_.forward_init();
    for (long i = 0; i < num_steps_; ++i) {
        check_cuda(cudaMalloc(&d_x_.at(i), features->at(0).size_ * sizeof(float)));
    }

    pipeline(true, num_chunks_);

    linear_self_.forward_free();
    for (long i = 0; i < num_steps_; ++i) {
        check_cuda(cudaFree(d_x_.at(i)));
    }

}

void SageLinearPipelined::backward_in(long chunk, long buffer) {

}

void SageLinearPipelined::backward_out(long chunk, long buffer) {

}

void SageLinearPipelined::backward_compute(long buffer) {

}

SageLinearGradientsChunked *SageLinearPipelined::backward(std::vector<Matrix<float>> *incoming_gradients) {

}
