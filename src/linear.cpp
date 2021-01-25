// Copyright 2020 Marcel Wagenländer

#include "linear.hpp"
#include "cuda_helper.hpp"
#include "dense_computation.hpp"
#include "tensors.hpp"

#include <chrono>
#include <cuda_runtime.h>
#include <random>


Linear::Linear() {}

Linear::Linear(CudaHelper *helper, long in_features, long out_features, long num_nodes) {
    set(helper, in_features, out_features, num_nodes);
}

void Linear::set(CudaHelper *helper, long in_features, long out_features, long num_nodes) {
    name_ = "linear";
    cuda_helper_ = helper;

    num_nodes_ = num_nodes;
    num_in_features_ = in_features;
    num_out_features_ = out_features;

    weight_.set(num_in_features_, num_out_features_, false);
    bias_.set(num_out_features_, 1, false);

    grad_weight_.set(weight_.num_rows_, weight_.num_columns_, false);
    grad_bias_.set(bias_.num_rows_, bias_.num_columns_, false);

    init_weight_bias();

    bias_expanded_.set(num_nodes, bias_.num_rows_, false);
    expand_bias();

    y_.set(num_nodes, num_out_features_, false);

    ones_ = std::vector<float>(num_nodes, 1.0);

    gradients_.set(num_nodes, in_features, false);
}

void Linear::init_weight_bias() {
    double k = 1.0 / static_cast<double>(num_in_features_);
    k = sqrt(k);
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_real_distribution<float> distr(-k, k);
    for (int i = 0; i < weight_.num_rows_ * weight_.num_columns_; ++i) {
        weight_.values_[i] = distr(generator);
    }
    for (int i = 0; i < bias_.num_rows_ * bias_.num_columns_; ++i) {
        bias_.values_[i] = distr(generator);
    }
}

std::vector<Matrix<float> *> Linear::get_parameters() {
    std::vector<Matrix<float> *> parameters(2);
    parameters[0] = &weight_;
    parameters[1] = &bias_;

    return parameters;
}

std::vector<Matrix<float> *> Linear::get_gradients() {
    std::vector<Matrix<float> *> gradients(2);
    gradients[0] = &grad_weight_;
    gradients[1] = &grad_bias_;

    return gradients;
}

std::vector<float *> Linear::get_gradients_cuda() {
    std::vector<float *> gradients(2);
    gradients[0] = d_dweight_;
    gradients[1] = d_db_;

    return gradients;
}

void Linear::set_gradients(Matrix<float> *weight_grads, Matrix<float> *bias_grads) {
    to_column_major_inplace(weight_grads);
    to_column_major_inplace(bias_grads);

    std::memcpy(grad_weight_.values_, weight_grads->values_, grad_weight_.size_ * sizeof(float));
    std::memcpy(grad_bias_.values_, bias_grads->values_, grad_bias_.size_ * sizeof(float));
}

void Linear::expand_bias() {
    for (long i = 0; i < bias_expanded_.num_columns_; ++i) {
        for (long j = 0; j < bias_expanded_.num_rows_; ++j) {
            bias_expanded_.values_[i * bias_expanded_.num_rows_ + j] = bias_.values_[i];
        }
    }
}

void Linear::allocate_gpu_memory_forward() {
    check_cuda(cudaMalloc(&d_weight_, weight_.size_ * sizeof(float)));
}

void Linear::free_gpu_memory_forward() {
    check_cuda(cudaFree(d_weight_));
}

void Linear::forward_init() {
    to_column_major_inplace(&weight_);
    to_column_major_inplace(&bias_);

    check_cuda(cudaMemcpy(d_weight_, weight_.values_, weight_.size_ * sizeof(float),
                          cudaMemcpyHostToDevice));
}

void Linear::forward_compute(float *d_x, long num_rows, float *d_y) {
    // needs to be reset at every call because it's overwritten with the result
    check_cuda(cudaMemcpy(d_y, bias_expanded_.values_, bias_expanded_.size_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    float alpha = 1.0;
    float beta = 1.0;
    check_cublas(cublasSgemm(cuda_helper_->cublas_handle,// PyTorch uses GEMM too
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             num_rows, num_out_features_, num_in_features_,
                             &alpha,
                             d_x, num_rows,
                             d_weight_, weight_.num_rows_,
                             &beta,
                             d_y, num_rows));
}

Matrix<float> *Linear::forward(Matrix<float> *x) {
    to_column_major_inplace(x);
    x_ = x;

    Linear::allocate_gpu_memory_forward();
    Linear::forward_init();

    float *d_x;
    check_cuda(cudaMalloc(&d_x, x->size_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_x, x->values_, x->size_ * sizeof(float),
                          cudaMemcpyHostToDevice));
    float *d_y;
    check_cuda(cudaMalloc(&d_y, bias_expanded_.size_ * sizeof(float)));

    Linear::forward_compute(d_x, num_nodes_, d_y);

    // get result of linear
    check_cuda(cudaMemcpy(y_.values_, d_y, y_.size_ * sizeof(float),
                          cudaMemcpyDeviceToHost));
    y_.is_row_major_ = false;

    Linear::free_gpu_memory_forward();

    return &y_;
}

void Linear::allocate_gpu_memory_backward() {
    check_cuda(cudaMalloc(&d_ones_, num_nodes_ * sizeof(float)));
    check_cuda(cudaMalloc(&d_db_, bias_expanded_.num_columns_ * sizeof(float)));
    check_cuda(cudaMalloc(&d_weight_, weight_.size_ * sizeof(float)));
    check_cuda(cudaMalloc(&d_dweight_, grad_weight_.size_ * sizeof(float)));
}

void Linear::free_gpu_memory_backward() {
    check_cuda(cudaFree(d_ones_));
    check_cuda(cudaFree(d_db_));
    check_cuda(cudaFree(d_weight_));
    check_cuda(cudaFree(d_dweight_));
}

void Linear::backward_init() {
    check_cuda(cudaMemcpy(d_ones_, ones_.data(), num_nodes_ * sizeof(float),
                          cudaMemcpyHostToDevice));
    check_cuda(cudaMemset(d_db_, 0, bias_expanded_.num_columns_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_weight_, weight_.values_, weight_.size_ * sizeof(float),
                          cudaMemcpyHostToDevice));
    check_cuda(cudaMemset(d_dweight_, 0, grad_weight_.size_ * sizeof(float)));
}

void Linear::backward_compute(float *d_dy, float *d_x, long num_rows, float *d_dx) {
    float alpha = 1.0;
    float beta = 1.0;

    // dBias = incoming_gradients * ones
    check_cublas(cublasSgemv(cuda_helper_->cublas_handle,
                             CUBLAS_OP_T,
                             num_rows, num_out_features_,
                             &alpha, d_dy, num_rows,
                             d_ones_, 1,
                             &beta, d_db_, 1));

    // dWeight = input.T * incoming_gradients
    check_cublas(cublasSgemm(cuda_helper_->cublas_handle,
                             CUBLAS_OP_T, CUBLAS_OP_N,
                             num_in_features_, num_out_features_, num_rows,
                             &alpha,
                             d_x, num_rows,
                             d_dy, num_rows,
                             &beta,
                             d_dweight_, grad_weight_.num_rows_));

    // gradients_input = incoming_gradients * weight.T
    beta = 0.0;
    check_cublas(cublasSgemm(cuda_helper_->cublas_handle,
                             CUBLAS_OP_N, CUBLAS_OP_T,
                             num_rows, weight_.num_rows_, num_out_features_,
                             &alpha,
                             d_dy, num_rows,
                             d_weight_, weight_.num_rows_,
                             &beta,
                             d_dx, num_rows));
}

void Linear::copy_gradients_to_cpu() {
    // gradients of bias
    check_cuda(cudaMemcpy(grad_bias_.values_, d_db_, grad_bias_.size_ * sizeof(float),
                          cudaMemcpyDeviceToHost));
    // gradients of weight
    check_cuda(cudaMemcpy(grad_weight_.values_, d_dweight_, grad_weight_.size_ * sizeof(float),
                          cudaMemcpyDeviceToHost));
}

Matrix<float> *Linear::backward(Matrix<float> *incoming_gradients) {
    to_column_major_inplace(incoming_gradients);
    to_column_major_inplace(x_);
    to_column_major_inplace(&weight_);
    to_column_major_inplace(&bias_);

    Linear::allocate_gpu_memory_backward();
    Linear::backward_init();

    float *d_dy;
    check_cuda(cudaMalloc(&d_dy, incoming_gradients->size_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_dy, incoming_gradients->values_, incoming_gradients->size_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    float *d_x;
    check_cuda(cudaMalloc(&d_x, x_->size_ * sizeof(float)));
    check_cuda(cudaMemcpy(d_x, x_->values_, x_->size_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    float *d_dx;
    check_cuda(cudaMalloc(&d_dx, x_->size_ * sizeof(float)));

    Linear::backward_compute(d_dy, d_x, num_nodes_, d_dx);

    check_cuda(cudaMemcpy(gradients_.values_, d_dx, gradients_.size_ * sizeof(float),
                          cudaMemcpyDeviceToHost));

    Linear::copy_gradients_to_cpu();
    Linear::free_gpu_memory_backward();

    return &gradients_;
}

// CHUNKED --- CHUNKED -- CHUNKED

LinearChunked::LinearChunked() {}

LinearChunked::LinearChunked(CudaHelper *helper, long chunk_size, long num_nodes, long num_in_features, long num_out_features) {
    set(helper, chunk_size, num_nodes, num_in_features, num_out_features);
}

LinearChunked::~LinearChunked() {
    if (keep_allocation_) {
        free_gpu_memory();
    }
}

void LinearChunked::set(CudaHelper *helper, long chunk_size, long num_nodes, long num_in_features, long num_out_features) {
    LinearChunked::set(helper, chunk_size, num_nodes, num_in_features, num_out_features, false);
}

void LinearChunked::set(CudaHelper *helper, long chunk_size, long num_nodes, long num_in_features, long num_out_features, bool keep_allocation) {
    name_ = "linear_chunked";
    cuda_helper_ = helper;
    chunk_size_ = chunk_size;
    num_in_features_ = num_in_features;
    num_out_features_ = num_out_features;
    keep_allocation_ = keep_allocation;

    num_chunks_ = ceil((float) num_nodes / (float) chunk_size_);
    if (num_chunks_ * chunk_size_ > num_nodes) {
        last_chunk_size_ = num_nodes - (num_chunks_ - 1) * chunk_size_;
    } else {
        last_chunk_size_ = chunk_size_;
    }

    if (num_chunks_ == 1) {
        linear_.set(cuda_helper_, num_in_features_, num_out_features_, last_chunk_size_);
    } else {
        linear_.set(cuda_helper_, num_in_features_, num_out_features_, chunk_size_);
    }

    y_ = std::vector<Matrix<float>>(num_chunks_);
    gradients_ = std::vector<Matrix<float>>(num_chunks_);
    long current_chunk_size = chunk_size_;
    for (int i = 0; i < num_chunks_; ++i) {
        if (i == (num_chunks_ - 1)) {
            current_chunk_size = last_chunk_size_;
        }
        y_.at(i).set(current_chunk_size, num_out_features, false);
        gradients_.at(i).set(current_chunk_size, num_in_features, false);
    }

    if (keep_allocation_) {
        allocate_gpu_memory();
    }
}

std::vector<Matrix<float> *> LinearChunked::get_parameters() {
    return linear_.get_parameters();
}

std::vector<Matrix<float> *> LinearChunked::get_gradients() {
    return linear_.get_gradients();
}

void LinearChunked::allocate_gpu_memory_forward() {
    check_cuda(cudaMalloc(&d_x_, chunk_size_ * num_in_features_ * sizeof(float)));
    check_cuda(cudaMalloc(&d_y_, y_.at(0).size_ * sizeof(float)));

    linear_.allocate_gpu_memory_forward();
}

void LinearChunked::allocate_gpu_memory_backward() {
    LinearChunked::allocate_gpu_memory();
}

void LinearChunked::allocate_gpu_memory() {
    long input_size = chunk_size_ * num_in_features_;
    check_cuda(cudaMalloc(&d_x_, input_size * sizeof(float)));
    check_cuda(cudaMalloc(&d_y_, chunk_size_ * num_out_features_ * sizeof(float)));
    check_cuda(cudaMalloc(&d_dx_, input_size * sizeof(float)));

    linear_.allocate_gpu_memory_backward();
}

void LinearChunked::free_gpu_memory_forward() {
    check_cuda(cudaFree(d_x_));
    check_cuda(cudaFree(d_y_));

    linear_.free_gpu_memory_forward();
}

void LinearChunked::free_gpu_memory_backward() {
    LinearChunked::free_gpu_memory();
}

void LinearChunked::free_gpu_memory() {
    check_cuda(cudaFree(d_x_));
    check_cuda(cudaFree(d_y_));
    check_cuda(cudaFree(d_dx_));

    linear_.free_gpu_memory_backward();
}

std::vector<Matrix<float>> *LinearChunked::forward(std::vector<Matrix<float>> *x) {
    x_ = x;

    if ((long) x->size() != num_chunks_) {
        throw "Input has wrong number of chunks";
    }
    for (int i = 0; i < num_chunks_; ++i) {
        to_column_major_inplace(&x->at(i));
    }

    if (!keep_allocation_) {
        allocate_gpu_memory_forward();
    }

    linear_.forward_init();

    for (int i = 0; i < num_chunks_; ++i) {
        // in
        check_cuda(cudaMemcpy(d_x_, x->at(i).values_, x->at(i).size_ * sizeof(float),
                              cudaMemcpyHostToDevice));

        // compute
        linear_.forward_compute(d_x_, x->at(i).num_rows_, d_y_);

        // out
        check_cuda(cudaMemcpy(y_.at(i).values_, d_y_, y_.at(i).size_ * sizeof(float),
                              cudaMemcpyDeviceToHost));
        y_.at(i).is_row_major_ = false;
    }

    if (!keep_allocation_) {
        free_gpu_memory_forward();
    }

    return &y_;
}

std::vector<Matrix<float>> *LinearChunked::backward(std::vector<Matrix<float>> *incoming_gradients) {
    if (y_.size() != incoming_gradients->size()) {
        throw "Output and incoming gradients have a different number of chunks";
    }
    for (int i = 0; i < num_chunks_; ++i) {
        to_column_major_inplace(&incoming_gradients->at(i));
    }

    if (!keep_allocation_) {
        allocate_gpu_memory_backward();
    }

    linear_.backward_init();

    for (int i = 0; i < num_chunks_; ++i) {
        // in
        check_cuda(cudaMemcpy(d_y_, incoming_gradients->at(i).values_, incoming_gradients->at(i).size_ * sizeof(float),
                              cudaMemcpyHostToDevice));
        check_cuda(cudaMemcpy(d_x_, x_->at(i).values_, x_->at(i).size_ * sizeof(float),
                              cudaMemcpyHostToDevice));

        // compute
        linear_.backward_compute(d_y_, d_x_, incoming_gradients->at(i).num_rows_, d_dx_);

        // out
        check_cuda(cudaMemcpy(gradients_.at(i).values_, d_dx_, gradients_.at(i).size_ * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }

    linear_.copy_gradients_to_cpu();

    if (!keep_allocation_) {
        free_gpu_memory_backward();
    }

    return &gradients_;
}

// PIPELINED --- PIPELINED --- PIPELINED

LinearPipelined::LinearPipelined() {}

LinearPipelined::LinearPipelined(CudaHelper *helper, long chunk_size, long num_nodes, long num_in_features, long num_out_features) {
    set(helper, chunk_size, num_nodes, num_in_features, num_out_features);
}

void LinearPipelined::set(CudaHelper *helper, long chunk_size, long num_nodes, long num_in_features, long num_out_features) {
    LinearChunked::set(helper, chunk_size, num_nodes, num_in_features, num_out_features);

    name_ = "linear_pipelined";
    num_steps_ = 2;

    d_x_ = std::vector<float *>(num_steps_);
    d_y_ = std::vector<float *>(num_steps_);
    d_dy_ = std::vector<float *>(num_steps_);
    d_dx_ = std::vector<float *>(num_steps_);
}

void LinearPipelined::forward_in(long chunk, long buffer) {
    check_cuda(cudaMemcpyAsync(d_x_.at(buffer), x_->at(chunk).values_, x_->at(chunk).size_ * sizeof(float),
                               cudaMemcpyHostToDevice, cuda_helper_->stream_in_));
}

void LinearPipelined::forward_out(long chunk, long buffer) {
    check_cuda(cudaMemcpyAsync(y_.at(chunk).values_, d_y_.at(buffer), y_.at(chunk).size_ * sizeof(float),
                               cudaMemcpyDeviceToHost, cuda_helper_->stream_out_));
    y_.at(chunk).is_row_major_ = false;
}

void LinearPipelined::forward_compute(long chunk, long buffer) {
    linear_.forward_compute(d_x_.at(buffer), x_->at(chunk).num_rows_, d_y_.at(buffer));
}

std::vector<Matrix<float>> *LinearPipelined::forward(std::vector<Matrix<float>> *x) {
    x_ = x;

    if ((long) x->size() != num_chunks_) {
        throw "Input has wrong number of chunks";
    }
    for (long i = 0; i < num_chunks_; ++i) {
        to_column_major_inplace(&x->at(i));
    }

    linear_.allocate_gpu_memory_forward();
    linear_.forward_init();
    for (long i = 0; i < num_steps_; ++i) {
        check_cuda(cudaMalloc(&d_x_.at(i), x->at(0).size_ * sizeof(float)));
        check_cuda(cudaMalloc(&d_y_.at(i), y_.at(0).size_ * sizeof(float)));
    }

    pipeline(true, num_chunks_);

    // free
    linear_.free_gpu_memory_forward();
    for (long i = 0; i < num_steps_; ++i) {
        check_cuda(cudaFree(d_x_.at(i)));
        check_cuda(cudaFree(d_y_.at(i)));
    }

    return &y_;
}

void LinearPipelined::backward_in(long chunk, long buffer) {
    check_cuda(cudaMemcpyAsync(d_dy_.at(buffer), incoming_gradients_->at(chunk).values_, incoming_gradients_->at(chunk).size_ * sizeof(float),
                               cudaMemcpyHostToDevice, cuda_helper_->stream_in_));
    check_cuda(cudaMemcpyAsync(d_x_.at(buffer), x_->at(chunk).values_, x_->at(chunk).size_ * sizeof(float),
                               cudaMemcpyHostToDevice, cuda_helper_->stream_in_));
}

void LinearPipelined::backward_out(long chunk, long buffer) {
    check_cuda(cudaMemcpyAsync(gradients_.at(chunk).values_, d_dx_.at(buffer), gradients_.at(chunk).size_ * sizeof(float),
                               cudaMemcpyDeviceToHost, cuda_helper_->stream_out_));
}

void LinearPipelined::backward_compute(long chunk, long buffer) {
    linear_.backward_compute(d_dy_.at(buffer), d_x_.at(buffer), incoming_gradients_->at(chunk).num_rows_, d_dx_.at(buffer));
}

std::vector<Matrix<float>> *LinearPipelined::backward(std::vector<Matrix<float>> *incoming_gradients) {
    incoming_gradients_ = incoming_gradients;

    if (y_.size() != incoming_gradients->size()) {
        throw "Output and incoming gradients have a different number of chunks";
    }
    for (long i = 0; i < num_chunks_; ++i) {
        to_column_major_inplace(&incoming_gradients->at(i));
    }

    for (long i = 0; i < num_steps_; ++i) {
        check_cuda(cudaMalloc(&d_dy_.at(i), incoming_gradients->at(0).size_ * sizeof(float)));
        check_cuda(cudaMalloc(&d_x_.at(i), x_->at(0).size_ * sizeof(float)));
        check_cuda(cudaMalloc(&d_dx_.at(i), x_->at(0).size_ * sizeof(float)));
    }

    linear_.allocate_gpu_memory_backward();
    linear_.backward_init();

    pipeline(false, num_chunks_);

    linear_.copy_gradients_to_cpu();
    linear_.free_gpu_memory_backward();
    for (long i = 0; i < num_steps_; ++i) {
        check_cuda(cudaFree(d_dy_.at(i)));
        check_cuda(cudaFree(d_x_.at(i)));
        check_cuda(cudaFree(d_dx_.at(i)));
    }

    return &gradients_;
}
